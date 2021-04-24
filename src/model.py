import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# ML architecture

# class Net(nn.Module):
    # num_classes = 1
    # def __init__(self,  num_classes):
        # super(Net, self).__init__()       
        # self.conv1 = nn.Conv1d(in_channels=54, out_channels=100, kernel_size=3, stride=2, padding=1)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv1_bn = nn.BatchNorm1d(100)
        
        # self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight)
        # self.conv2_bn = nn.BatchNorm1d(100)
        
        # self.fc1 = nn.Linear(2600, num_classes)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        
    # def forward(self, x):      
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.conv1_bn(x)
        
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.conv2_bn(x)
        
        # x = x.view(x.size(0), -1)
        # x = torch.sigmoid(self.fc1(x))
        
        # return x


class LSTM_Encoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout):
        super(LSTM_Encoder, self).__init__()
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids (+ padding = 1 ?)
        self.embedding = nn.Embedding(20, 
                embedding_dim)
                # padding_idx=0
        # RNN - LSTM
        self.lstm = nn.LSTM(embedding_dim,
                lstm_dim,
                num_layers=2, 
                batch_first=True,
                dropout=dropout)

    def init_hidden(self, batch_size, device):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(2, 
                        batch_size, 
                        self.lstm_dim).to(device)),
                    autograd.Variable(torch.zeros(2,
                        batch_size,
                        self.lstm_dim)).to(device))
        else:
            return (autograd.Variable(torch.zeros(2,
                        batch_size,
                        self.lstm_dim)),
                    autograd.Variable(torch.zeros(2,
                        batch_size,
                        self.lstm_dim)))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        device = padded_embeds.device
        # Before using PyTorch pack_padded_sequence 
        # we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds,
                lengths,
                batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size, device)
        # Feed into the RNN
        lstm.flatten_parameters()
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, seq, lengths):
        # Encoder:
        # Embedding
        embeds = self.embedding(seq)
        # LSTM Acceptor
        lstm_out = self.lstm_pass(self.lstm, embeds, lengths)
        last_cell = torch.cat([lstm_out[i, j.data - 1]
            for i, j in enumerate(lengths)]).view(len(lengths), 
                                                  self.lstm_dim)
        return last_cell


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.dataset = hparams.dataset
        self.lr = hparams.lr # Learning rate
        self.wd = hparams.wd # Weight decay
        self.dropout_rate = hparams.dropout
        self.lstm_dim = hparams.lstm_dim
        self.aa_embedding_dim = hparams.aa_embedding_dim
        self.mhc_embedding_dim = hparams.mhc_embedding_dim
        # TCR Encoder
        self.tcr_encoder = LSTM_Encoder(self.aa_embedding_dim,
                                        self.lstm_dim, 
                                        dropout)
        self.encoding_dim = self.lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(self.aa_embedding_dim,
                                        self.lstm_dim,
                                        self.dropout_rate)
        # MHC Embedding
        self.mhc_embedding = nn.Embedding(self.aa_embedding_dim,
                                          self.mhc_embedding_dim)
                                          # padding_idx=0)
        # MLP input size depends on model encoders
        self.mlp_dim = self.lstm_dim \
                + self.encoding_dim \
                + self.mhc_embedding_dim
        # MLP
        mlp_dim_sqrt = int(sqrt(np.sqrt(self.mlp_dim)))
        self.hidden_layer = nn.Linear(self.mlp_dim,
                                      mlp_dim_sqrt)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(mlp_dim_sqrt, 1)
        self.dropout = nn.Dropout(p = self.dropout_rate)

    def forward(self, tcr_batch, pep_batch, mhc_batch):
        # TODO: Include energy values
        tcr_encoding = self.tcr_encoder(*tcr_batch)
        pep_encoding = self.pep_encoder(*pep_batch)
        mhc_embedding = self.mhc_embedding(mhc_batch)
        mlp_input = [
            pep_encoding,
            tcr_encoding,
            mhc_embedding
        ]
        # MLP
        concat = torch.concat(mlp_input, 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        tcr, pep, mhc, y, weight = batch
        len_tcr = torch.sum((tcr > 0).int(), dim = 1)
        len_pep = torch.sum((pep > 0).int(), dim = 1)
        tcr_batch = (tcr, len_tcr) # LSTM
        pep_batch = (pep, len_pep)
        y_hat = self.forward(tcr_batch, pep_batch, mhc).squeeze()
        return y, y_hat, weight

    def training_step(self, batch, batch_idx):
        self.train()
        y, y_hat, weight = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        self.eval()
        step = self.step(batch)
        if !step:
            return None
        y, y_hat, _ = step
        return {
            'val_loss': F.binary_cross_entropy(y_hat.view(-1, 1),
                                               y.view(-1, 1)),
            'y_hat': y_hat, 
            'y': y
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'].view(-1, 1) for x in outputs])
        y_hat = torch.cat([x['y_hat'].view(-1, 1) for x in outputs])
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), 
                            y_hat.detach().cpu().numpy())
        print(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
        return {
            'avg_val_loss': avg_loss,
            'val_auc': auc, 
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr = self.lr, 
                                weight_decay = self.wd)
