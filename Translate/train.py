from load_data import MyDatasets, get_dataloader, causal_mask
import config
from Transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import torch

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

def train(dataset, model, optimizer, loss_fn):
    train_dataloader, eval_dataloader = get_dataloader(dataset,config.BATCH_SIZE,config.NUM_WORKERS, 0.8)
    model.train()
    for epoch in range(config.EPOCHS):
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
            print(f'Epoch {epoch}: loss = {loss.item():.2f}')

    model,eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            encoder_input = batch["encoder_input"].cuda()  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].cuda()  # (batch_size, 1, 1, seq_len). 
            #batch_size should be 1
            assert encoder_mask.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model,encoder_input,encoder_mask,dataset.tgt_word_dict,config.MAX_LENGTH)
            #Convert to text
            output = "".join([dataset.tgt_index_dict[w.item()] for w in model_output])

            print('---'*5)
            print(batch['src_text'])
            print(batch['tgt_text'])
            print(output)

            break

    torch.save(model,'Transformer.pth')




dataset = MyDatasets(config.TRAIN_FILE,config.MAX_LENGTH)
model = Transformer(dataset.src_vocab_size, dataset.tgt_vocab_size, config.D_MODEL, config.D_FFN, config.N, config.HEADS, config.DROPOUT_PROB).cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=config.LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
train(dataset,model,optimizer,loss_fn)