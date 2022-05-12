import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd


# if a word is not in vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc.reset_index(drop=True)

    def __len__(self):
        """ Length of the dataset """
        
        L = self.df.shape[0]
        L = L-4 #first 2 and last 2 wont be used
        
        
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        
        x = np.array(self.df.loc[idx:idx+4,  'word'])
        y = self.df.loc[idx+2, 'label']
        
        return x, y 


def label_encoding(cat_arr):
   """ Given a numpy array of strings returns a dictionary with label encodings.

   """
   
   cat_arr = cat_arr.astype('str')
   unique_strings = np.unique(cat_arr)

   vocab2index = {o:i for i,o in enumerate(unique_strings)}
   
   
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()

    df_enc["word"] = df_enc["word"].apply(lambda x: vocab2index.get(x,V))
    df_enc["label"] = df_enc["label"].apply(lambda x: label2index[x])


    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)

        self.emb1 = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(5*emb_size, n_class)


        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """


        x = self.emb1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        

        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):

        losses = []
        model.train()
        for x,y in train_dl:
            y_hat = model(torch.LongTensor(x))
            loss = F.cross_entropy(y_hat, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = np.mean(losses)

        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):

    losses = []
    preds = []
    y_true = []
    for x,y in valid_dl:
        model.eval()
        y_hat = model(torch.LongTensor(x))
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=-1)
        losses.append(loss.item())
        preds.extend(np.argmax(y_hat.detach().numpy(),axis=1))
        y_true.extend(y.numpy())


    val_loss = np.mean(losses)
    val_acc = accuracy_score(y_true, preds)

    return val_loss, val_acc

