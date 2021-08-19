# -*- coding: utf-8 -*-

 
import argparse
import torch
from torch import nn
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import os

 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
 
train_novel_path ='/content/drive/MyDrive/RNN/town.txt'
model_save_path = "/content/drive/MyDrive/RNN/model623.pkl"
save_novel_path ="/content/drive/MyDrive/RNN/novel.txt"


use_gpu =torch.cuda.is_available()
print('torch.cuda.is_available() == ',use_gpu)
device = torch.device('cuda:0')

#build dictionary and string
word_to_id=dict()
id_to_word=dict()
essay=str()

#get the dictionary of word and id, record the training essay
def get_w2i(file):
  global word_to_id
  global essay
  with open(file,encoding='UTF-8') as f:
    essay=f.read()
  for word in essay:
    if word not in word_to_id:
      word_to_id[word]=len(word_to_id)


#build mydataset  
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,args,):
        self.args = args
        self.words = [word_to_id[w] for w in essay]
        self.vocab_num = len(word_to_id)
        
    def __len__(self):
        return len(self.words) - self.args.sequence_length
 
    def __getitem__(self, index):
        # if i have a sentence like "you are my shine"
        # then output is  'you are my'   'are my shine'
        return torch.tensor(self.words[index:index+self.args.sequence_length]).cuda(),torch.tensor(self.words[index+1:index+self.args.sequence_length+1]).cuda()


#bulid model
class myModel(nn.Module):
    def __init__(self, dataset):
        super(myModel, self).__init__()
        self.input_size = 128
        self.hidden_size = 256
        self.embedding_dim = self.input_size
        self.num_layers = 2
 
        n_vocab = dataset.vocab_num
        
        #change the word to word vector
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        
        #find out the hidden information
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,            
        )
        self.rnn.cuda()
        
        #make sure the output which has the shape(n_vocab)
        self.linear = nn.Linear(self.hidden_size, n_vocab).cuda() 
        
    def forward(self, x, prev_state):
        
        embed = self.embedding(x).cuda()
        output,state = self.rnn(embed, prev_state)
        logits = self.linear(output)
 
        return logits,state
 
    def init_state(self, sequence_length):
        #get the hidden init
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda())

def predict(dataset, model, text, next_words=20):
    #generate the essay with the length of next_words
    words = list(text)
    model.eval()
 
    state= model.init_state(len(words))
 
    for i in range(0, next_words):
        x = torch.tensor([[word_to_id[w] for w in words[i:]]])
        y_pred, state = model(x, state)
 
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(id_to_word[word_index])
 
    return "".join(words)

def mytrain(dataset, model, args):
    model.to(device)
    model.train()
    
    #divide the batch
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )
 
    #define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #the loss of  last epoch
    last_loss=np.inf
    
    #training
    for epoch in range(args.max_epochs):
        state = model.init_state(args.sequence_length)
        total_loss=0
        for batch, (x, y) in enumerate(dataloader):
            
            optimizer.zero_grad()
            x = x.cuda()
            y= y.cuda()
            y_pred, state = model(x, state)
            
            loss = criterion(y_pred.transpose(1, 2), y)
            loss =loss.to(device)
            state = state.detach()            
 
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            
        if total_loss<last_loss:
          torch.save(model, model_save_path)
          last_loss=total_loss
        print({ 'epoch': epoch, 'loss': total_loss })    




# you can input the parameter by command line
parser = argparse.ArgumentParser(description='rnn')
parser.add_argument('--max-epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=1024)  ) 
parser.add_argument('--sequence-length', type=int, default=64)   
args = parser.parse_args([])
 

# load word_to_id, id_to_word, data, model
get_w2i(train_novel_path)
id_to_word={v:k for k,v in word_to_id.items()}
data=MyDataset(args)
model=myModel(data)
print(model)

#training
mytrain(data,model,args)

#testing
pred_novel_start_text='翠翠说' 
genessay=predict(data, model, pred_novel_start_text,300)
print(genessay)

     
    

