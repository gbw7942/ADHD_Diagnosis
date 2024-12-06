#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
Note: When training Transformer models on Chinese corpora, 
Chinese sentences are typically split character-by-character, 
without the need for word segmentation.

Note: Sequence length (seq_len) is consistent within the same batch, 
but may vary between different batches.
"""

import numpy as np
from langconv import Converter
from nltk import word_tokenize
from collections import Counter
import config
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence


def cht_to_chs(sent):
    """
    Convert traditional Chinese characters to simplified Chinese.
    
    Args:
        sent (str): Input sentence in traditional Chinese
    
    Returns:
        str: Sentence converted to simplified Chinese
    """
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent


class MyDatasets(Dataset):
    def __init__(self, data_path, max_seq_len):
        """
        Initialize the dataset for machine translation.
        
        Args:
            data_path (str): Path to the parallel text file
            max_seq_len (int): Maximum allowed sequence length
        """
        self.max_seq_len = max_seq_len

        # Define special token tensors
        self.PAD = torch.tensor([0], dtype=torch.int64)  # Padding token
        self.BOS = torch.tensor([1], dtype=torch.int64)  # Beginning of Sentence token
        self.EOS = torch.tensor([2], dtype=torch.int64)  # End of Sentence token

        # Load data and tokenize
        self.data_src, self.data_tgt = self.load_data(data_path)
        
        # Build vocabularies (word2index, vocab_size, index2word)
        self.src_word_dict, self.src_vocab_size, self.src_index_dict = self.build_src_dict(self.data_src)
        self.tgt_word_dict, self.tgt_vocab_size, self.tgt_index_dict = self.build_tgt_dict(self.data_tgt)

    def load_data(self, path):
        """
        Read English and Chinese parallel text data.
        
        Tokenize each sample and construct word lists with start and end tokens.
        
        Expected format:
        en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        
        Args:
            path (str): Path to parallel text file
        
        Returns:
            tuple: Lists of tokenized English and Chinese sentences
        """
        en = []
        cn = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split("\t")
                sent_en = sent_en.lower()
                
                # Convert Chinese characters from traditional to simplified
                sent_cn = cht_to_chs(sent_cn)
                
                # Tokenize English sentence
                sent_en = word_tokenize(sent_en)
                
                # Split Chinese sentence into individual characters
                sent_cn = [char for char in sent_cn]
                
                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_src_dict(self, sentences, max_words=5e4):
        """
        Build vocabulary dictionary for source language.
        
        Args:
            sentences (list): List of tokenized source language sentences
            max_words (int): Maximum number of words to include in vocabulary
        
        Returns:
            tuple: Source word dictionary, vocabulary size, and index dictionary
        """
        # Count word frequencies in the dataset
        word_count = Counter([word for sent in sentences for word in sent])
        
        # Retain top max_words based on frequency
        # Add PAD token
        ls = word_count.most_common(int(max_words))
        src_vocab_size = len(ls) + 1
        src_word_dict = {w[0]: index + 1 for index, w in enumerate(ls)}
        src_word_dict['PAD'] = config.PAD
        
        # Create index to word mapping
        src_index_dict = {v: k for k, v in src_word_dict.items()}
        return src_word_dict, src_vocab_size, src_index_dict

    def build_tgt_dict(self, sentences, max_words=5e4):
        """
        Build vocabulary dictionary for target language.
        
        Args:
            sentences (list): List of tokenized target language sentences
            max_words (int): Maximum number of words to include in vocabulary
        
        Returns:
            tuple: Target word dictionary, vocabulary size, and index dictionary
        """
        # Count word frequencies in the dataset
        word_count = Counter([word for sent in sentences for word in sent])
        
        # Retain top max_words based on frequency
        # Add PAD, BOS, EOS tokens
        ls = word_count.most_common(int(max_words))
        tgt_vocab_size = len(ls) + 3
        tgt_word_dict = {w[0]: index + 3 for index, w in enumerate(ls)}
        tgt_word_dict['PAD'] = 0
        tgt_word_dict['BOS'] = 1
        tgt_word_dict['EOS'] = 2
        
        # Create index to word mapping
        tgt_index_dict = {v: k for k, v in tgt_word_dict.items()}
        return tgt_word_dict, tgt_vocab_size, tgt_index_dict

    def __getitem__(self, index):
        """
        Prepare a single data sample for model training.
        
        Converts words to indices, adds special tokens, and pads sequences.
        
        Args:
            index (int): Index of the sample in the dataset
        
        Returns:
            dict: Processed data sample with encoder/decoder inputs, masks, and labels
        """
        # Convert words to token indices
        enc_input_tokens = [self.src_word_dict[word] for word in self.data_src[index]]
        dec_input_tokens = [self.tgt_word_dict[word] for word in self.data_tgt[index]]

        # Calculate padding tokens needed
        # Subtract 2 to account for BOS and EOS tokens
        enc_num_padding_tokens = self.max_seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_seq_len - len(dec_input_tokens) - 1

        # Validate sequence length
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Construct encoder input with special tokens and padding
        encoder_input = torch.cat(
            [
                self.BOS,  # Add beginning of sentence token
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.EOS,  # Add end of sentence token
                self.PAD.repeat(enc_num_padding_tokens),  # Add padding
            ],
            dim=0,
        )

        # Construct decoder input with special tokens
        decoder_input = torch.cat(
            [
                self.BOS,  # Add beginning of sentence token
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.PAD.repeat(dec_num_padding_tokens),  # Add padding
            ],
            dim=0,
        )

        # Construct decoder output with target tokens and special tokens
        decoder_output = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.EOS,  # Add end of sentence token
                self.PAD.repeat(dec_num_padding_tokens),  # Add padding
            ],
            dim=0,
        )

        # Validate tensor lengths
        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert decoder_output.size(0) == self.max_seq_len

        return {
            "encoder_input": encoder_input,  # Input for encoder (seq_len)
            "decoder_input": decoder_input,  # Input for decoder (seq_len)
            "encoder_mask": (encoder_input != self.PAD).unsqueeze(0).unsqueeze(0).int(),  # Mask to ignore padding (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.PAD).unsqueeze(0).int() & causal_mask(
                decoder_output.size(0)),  # Combined padding and causal masking
            "label": decoder_output,  # Target output sequence (seq_len)
            "src_text": self.data_src[index] + [" "] * (enc_num_padding_tokens + 2),  # Original source text with padding
            "tgt_text": self.data_tgt[index] + [" "] * (dec_num_padding_tokens + 1),  # Original target text with padding
        }
        # In masks: False indicates tokens to be ignored, True indicates tokens to be preserved

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data_src)


def causal_mask(size):
    """
    Create a causal (triangular) mask to prevent attending to future tokens.
    
    Args:
        size (int): Size of the sequence
    
    Returns:
        torch.Tensor: Causal mask tensor
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_dataloader(datasets, batch_size, num_workers, ratio):
    """
    Create train and validation dataloaders.
    
    Args:
        datasets (Dataset): Full dataset
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        ratio (float): Proportion of data to use for training
    
    Returns:
        tuple: Training and validation dataloaders
    """
    # Calculate train and validation dataset sizes
    total_size = len(datasets)
    train_size = int(total_size * ratio)
    val_size = total_size - train_size

    # Split dataset into train and validation sets
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    # Create training dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomize data order in each epoch
        num_workers=num_workers,  # Use multiple subprocesses for data loading
    )

    # Create validation dataloader
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # Process one sample at a time during validation
        shuffle=False,  # Maintain original order during validation
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    # Example usage and testing
    dataset = MyDatasets(config.TRAIN_FILE, max_seq_len=30)
    x = dataset[0]
    print()