from load_data import MyDatasets, get_dataloader, causal_mask
import config
from Transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import torch
import nltk

def greedy_decode(model, input, input_mask, tgt_word_dict, max_len):
    bos_idx = tgt_word_dict['BOS']
    eos_idx = tgt_word_dict['EOS']

    encoder_output = model.encode(input,input_mask)
    decoder_input = torch.empty(1,1).fill_(bos_idx).type_as(input).cuda()
    while True:
        if decoder_input.size(1) == max_len:
            break
        # when decoder_input col len is smaller than max_len, continue to append mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(input_mask).cuda()

        output = model.decode(encoder_output, input_mask, decoder_input, decoder_mask)

        prob = model.project(output[:,-1])
        
        _, next_word = torch.max(prob, dim=1)
        #append the predicted wordto decoder input and move to next word
        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(input).fill_(next_word.item()).cuda()],dim=1)

        if next_word == eos_idx:
            break # reach the end of the sequence
    return decoder_input.unsqueeze(0)

def greedy_decode_with_penalty(model, input, input_mask, tgt_word_dict, max_len, repetition_penalty=1.2):
    bos_idx = tgt_word_dict['BOS']
    eos_idx = tgt_word_dict['EOS']
    encoder_output = model.encode(input, input_mask)
    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(input).cuda()
    generated_tokens = []

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(input_mask).cuda()
        output = model.decode(encoder_output, input_mask, decoder_input, decoder_mask)
        prob = model.project(output[:, -1])

        # Apply repetition penalty
        for token in generated_tokens:
            prob[0, token] /= repetition_penalty

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_word.item()).type_as(input).cuda()], dim=1)
        generated_tokens.append(next_word.item())

        if next_word == eos_idx:
            break
    return decoder_input.unsqueeze(0)


def train(dataset, model, optimizer, loss_fn):
    loss_total=0
    train_dataloader, eval_dataloader = get_dataloader(dataset,config.BATCH_SIZE,config.NUM_WORKERS, 0.8)
    model.train()
    for epoch in range(config.EPOCHS):
        loss_total=0
        for batch in train_dataloader:
            encoder_input = batch['encoder_input'].cuda()  # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].cuda()    # (batch_size, 1, 1, seq_len)
            decoder_input = batch['decoder_input'].cuda()  # (batch_size, seq_len)
            decoder_mask = batch['decoder_mask'].cuda()    # (batch_size, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input,encoder_mask)  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (batch_size, seq_len, d_model)
            project_output = model.project(decoder_output)  # (batch_size, seq_len, vocab_size)

            label = batch['label'].cuda()    # (batch_size, seq_len)

            loss = loss_fn(project_output.view(-1, dataset.tgt_vocab_size),label.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            loss_total+=loss.item()
        print(f'Epoch {epoch}: loss = {loss_total/(len(dataset)*0.8):.2f}')
        

    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            encoder_input = batch["encoder_input"].cuda()  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].cuda()  # (batch_size, 1, 1, seq_len). 
            #batch_size should be 1
            assert encoder_mask.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model,encoder_input,encoder_mask,dataset.tgt_word_dict,config.MAX_LENGTH)
            print(model_output.shape)
            #Convert to text
            output = "".join([dataset.tgt_index_dict[w.item()] for w in model_output[0][0][:]])

            print('---'*5)
            print(batch['src_text'])
            print(batch['tgt_text'])
            print(output)

            break

    torch.save(model,'Transformer.pth')


# def predict(model_path,input_sentence,dataset, max_len=config.MAX_LENGTH):
#     model = torch.load(model_path).cuda()
#     model.eval()
#     with torch.no_grad():
#             # Tokenize and convert the input sentence to tensor
#         input_tokens = [dataset.src_word_dict.get(word) for word in nltk.word_tokenize(input_sentence)]
#         input_tensor = torch.tensor(input_tokens).unsqueeze(0).cuda()  # Add batch dimension

#         # Create input mask
#         input_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2).type(torch.float32).cuda()
#         assert input_mask.size(0) == 1, "Batch size must be 1 for validation"
#         output_tensor = greedy_decode(model,input_tensor,input_mask,dataset.tgt_word_dict,max_len)
#         output = "".join([dataset.tgt_index_dict[w.item()] for w in output_tensor[0][0][:]])


#     return output, output_tensor

def predict(model_path, input_sentence, dataset, max_len=config.MAX_LENGTH):  
    model = torch.load(model_path)#.cuda()  
    model.eval()  
    with torch.no_grad():  
        # Tokenize and convert the input sentence to tensor  
        input_tokens = []  
        for word in nltk.word_tokenize(input_sentence.lower()):  
            token = dataset.src_word_dict.get(word)  
            if token is not None:  
                input_tokens.append(token)  
            else:  
                print(f"Warning: '{word}' not found in source vocabulary. It will be ignored.")  
        
        if not input_tokens:  
            raise ValueError("Input sentence contains no valid tokens.")  

        input_tensor = torch.tensor(input_tokens).unsqueeze(0).cuda()  # Add batch dimension 

        # Create input mask  
        input_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2).type(torch.float32)#.cuda() 
        assert input_mask.size(0) == 1, "Batch size must be 1 for validation"  
        output_tensor = greedy_decode_with_penalty(model, input_tensor, input_mask, dataset.tgt_word_dict, config.MAX_LENGTH)  
        output = "".join([dataset.tgt_index_dict[w.item()] for w in output_tensor[0][0][:]])  

    return output, output_tensor


dataset = MyDatasets(config.TRAIN_FILE,config.MAX_LENGTH)
model = Transformer(dataset.src_vocab_size, dataset.tgt_vocab_size, config.D_MODEL, config.D_FFN, config.N, config.HEADS, config.DROPOUT_PROB)#.cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=config.LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
# train(dataset,model,optimizer,loss_fn)
input_sentence = "i am playing baseball."
result, tensor = predict("Transformer.pth",input_sentence,dataset)

print(result,tensor)