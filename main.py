import sys
import os
from torchtext import data, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='experiment', help='experiment name')
parser.add_argument('--d_embed', type=int, default=30, help='embedding dimension')
parser.add_argument('--hidden_size', type=int, default=20, help='hidden size dimension')
parser.add_argument('--e_dropout', type=float, default=0.3, help='encoder embedding dropout')
parser.add_argument('--d_dropout', type=float, default=0.3, help='decoder embedding dropout')
parser.add_argument('--n_layers', type=int, default=2, help='number of encoder LSTM layers')
parser.add_argument('--bidirectional', action='store_true', help='bidirectional encoder LSTM')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--n_epochs', type=int, default=50000, help='maximum epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--tf_ratio', type=float, default=0.1, help='teacher forcing ratio')

opt = parser.parse_args()
print(opt)

max_seq_length = 30

def char_tokenizer(x):
    return list(x)

def get_data(filename):
    filename = filename.strip("data/").strip(".tsv")
    
    #Character level tokenization
    char_field = data.Field(tokenize=char_tokenizer, eos_token='<eos>', fix_length=max_seq_length)

    train_char = data.TabularDataset(path=f"data/{filename}_train.tsv",
                                  format='tsv',
                                  fields=[('prompt', char_field),
                                          ('answer', char_field)])
    val_char = data.TabularDataset(path=f"data/{filename}_val.tsv",
                                  format='tsv',
                                  fields=[('prompt', char_field),
                                          ('answer', char_field)])
    test_char = data.TabularDataset(path="data/test.tsv",
                                  format='tsv',
                                  fields=[('prompt', char_field),
                                          ('answer', char_field)])
    char_field.build_vocab(train_char,val_char,test_char)
    
    print(train_char.examples[0].prompt, train_char.examples[0].answer)
    print(val_char.examples[0].prompt, val_char.examples[0].answer)
    print(test_char.examples[0].prompt, test_char.examples[0].answer)
        
    print(f"Character field vocab size: {len(char_field.vocab)}")
    
    return char_field, train_char, val_char, test_char

char_field, train_char, val_char, test_char = get_data('data/pokemon.tsv')

def sort_key(ex):
    return len(ex.prompt)

def early_stop(val_acc_history, t=3, required_progress=0.01):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should early stop
    """

    if len(val_acc_history) < t + 1:
        return False
    else:
        first = np.array(val_acc_history[-t - 1:-1])
        second = np.array(val_acc_history[-t:])

        if np.all((second - first) < required_progress):
            return True
        else:
            return False

def evaluate_char(iterator, model, criterion, attention=False):
    model.eval()
    n_correct, n_pad, eval_losses = 0, 0, []
    iterator.init_epoch()
    for batch in iterator:
        if attention:
            out, attentions = model(batch)
        else:
            out = model(batch)
            
        n_correct += (torch.max(out,1)[1].data == batch.answer.transpose(0,1).data).sum()
        n_pad += (batch.answer.transpose(0,1).data == 1).sum()
        
        loss = 0          
        for i in range(max_seq_length):
            loss = criterion(out[:,:,i], batch.answer[i,:])
        eval_losses.append(loss)
    eval_loss = sum(eval_losses) / len(eval_losses)
    
    return n_correct, n_pad, eval_loss

def train_char_model(save_pt, attention=False):
    best_val_acc = 0.0
    
    for epoch in range(1, opt.n_epochs + 1):
        train_iter.init_epoch()

        for batch in train_iter:
            model.train()
            optimizer.zero_grad()
            
            if attention:
                out, attentions = model(batch)
            else:
                out = model(batch)
            
            loss = 0
            
            for i in range(max_seq_length):
                loss += criterion(out[:,:,i], batch.answer[i,:])
                
            loss.backward()
            clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), 10)
            optimizer.step()

        train_correct, train_pad, train_loss = evaluate_char(train_iter, model, criterion, attention)
        train_accuracy = 100 * (train_correct - train_pad) / (len(train_char)*max_seq_length - train_pad)
        val_correct, val_pad, val_loss = evaluate_char(val_iter, model, criterion, attention)
        val_accuracy = 100 * (val_correct - val_pad) / (len(val_char)*max_seq_length - val_pad)
        val_acc_history.append(val_accuracy)

        stop_training = early_stop(val_acc_history)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_pt)

        print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.2f}, Val Acc: {:.2f}, Best Val Acc: {:.2f}'.\
                format(epoch,
                       train_loss.data.numpy()[0],
                       val_loss.data.numpy()[0],
                       train_accuracy,
                       val_accuracy,
                       best_val_acc))

        if epoch % 100 == 0 or (epoch % 10 == 0 and epoch < 100):
            for batch in test_iter:
                if attention:
                    out, attentions = model(batch)
                else:
                    out = model(batch)
                    
                test_predictions.append([epoch,out.data.numpy()[0]])

        if early_stopping and stop_training or (train_correct == len(train_char)*max_seq_length):
            print('Early stop triggered.')
            break

def get_masks(input, pad_token=1):
    return (input != pad_token).float()

def softmask(input, mask):
    """
    input is of dims batch_size x seq_length
    mask if of dims batch_size x seq_length
    """

    exp_input = torch.exp(input)
    divisors = torch.sum(torch.mul(exp_input, mask.transpose(0,1)),dim=1)
    masked = torch.mul(exp_input, mask.transpose(0,1))

    return masked.div(divisors.unsqueeze(1).expand_as(masked))

class charLSTM_attn_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout, bidirectional):
        super(charLSTM_attn_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        if self.bidirectional:
            self.LSTM = nn.LSTMCell(self.hidden_size*3, self.hidden_size)
        else:
            self.LSTM = nn.LSTMCell(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, previous_char, hidden, cell_state, context, masks):
        embedded = self.embedding(previous_char)
        embedded = self.dropout(embedded)
        
        attn_weights = softmask(self.attn(torch.cat((embedded, hidden), 1)), masks)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),context)      

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)

        hidden, cell_state = self.LSTM(output, (hidden, cell_state))
        output = F.log_softmax(self.out(hidden), dim=1)

        return output, hidden, cell_state, attn_weights

    def init_CellHidden(self,batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        return result, result

class charLSTM_attn(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_embed,
                 hidden_size,
                 n_layers,
                 e_dropout,
                 d_dropout,
                 bidirectional,
                 max_length,
                 teacher_forcing_ratio):
        super(charLSTM_attn, self).__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.e_dropout = e_dropout
        self.d_dropout = d_dropout
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.teacher_forcing = True
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        self.embed = nn.Embedding(self.vocab_size, self.d_embed)
        self.encoder = nn.LSTM(input_size = self.d_embed,
                               hidden_size = self.hidden_size,
                               num_layers = self.n_layers,
                               batch_first = False,
                               dropout = self.e_dropout,
                               bidirectional = self.bidirectional)
        self.decoder = charLSTM_attn_decoder(hidden_size = self.hidden_size,
                       output_size = self.vocab_size,
                       max_length = self.max_length,
                       dropout = self.d_dropout,
                       bidirectional = self.bidirectional)
        self.encoder_dropout = nn.Dropout(p=e_dropout)
        self.decoder_dropout = d_dropout
        
    def teacher_forcing_on(self):
        self.teacher_forcing = True        
        
    def teacher_forcing_off(self):
        self.teacher_forcing = False
        
    def forward(self, batch):
        
        batch_size = batch.batch_size
        
        masks = get_masks(batch.prompt)
        x = self.embed(batch.prompt)
        x = self.encoder_dropout(x)
        
        state_hist, states = self.encoder(x)
        context = state_hist.transpose(0,1).contiguous()
        
        decoded = Variable(torch.FloatTensor(batch_size, self.vocab_size, self.max_length).zero_())
        attentions = Variable(torch.FloatTensor(batch_size, self.max_length, self.max_length).zero_())

        #make SOS token the next token after vocab_size
        decoder_input = Variable(torch.LongTensor(batch_size).fill_(self.vocab_size-1))
        decoder_cell, decoder_hidden = self.decoder.init_CellHidden(batch_size)
        
        for dim in range(self.max_length):
            decoder_output, decoder_cell, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                           decoder_hidden,
                                                                                           decoder_cell,
                                                                                           context,
                                                                                           masks)
            decoded[:,:,dim] = decoder_output
            attentions[:,:,dim] = decoder_attention
            
            if self.teacher_forcing and np.random.rand() < self.teacher_forcing_ratio:
                decoder_input = batch.answer[dim,:]
            else:
                decoder_input = torch.max(decoder_output,dim=1)[1]  
                        
        return decoded, attentions

train_iter, val_iter, test_iter = data.Iterator.splits((train_char,val_char,test_char),
                                                       batch_size=opt.batch_size,
                                                       sort_key=sort_key,
                                                       device=-1)
train_iter.repeat = False
early_stopping = False
n_embed = len(char_field.vocab)
criterion =  nn.NLLLoss()

model = charLSTM_attn(n_embed+1,
                      opt.d_embed,
                      opt.hidden_size,
                      opt.n_layers,
                      opt.e_dropout,
                      opt.d_dropout,
                      opt.bidirectional,
                      max_seq_length,
                      opt.tf_ratio)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

test_predictions = []
val_acc_history = []

train_char_model('saved_models/{0}.pt'.format(opt.experiment), True)

