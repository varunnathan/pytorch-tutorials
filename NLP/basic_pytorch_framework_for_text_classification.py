import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.datasets import SequenceTaggingDataset


TEXT = Field(tokenize='spacy', include_lengths=True)
LABEL = LabelField(dtype=torch.float)

# torchtext datasets
fields = [('text', TEXT), ('label', LABEL)]
train_ds = TabularDataset(path='', format='csv', fields=fields, skip_header=False)

# split train_ds into train and test
train_ds, val_ds = train_ds.split(split_ratio=0.7, random_state=random.seed(1))
print(vars(train_ds.examples[0]))

# build vocabulary
MAX_VOCAB = 30000
TEXT.build_vocab(train_ds, max_size=MAX_VOCAB)
LABEL.build_vocab(train_ds)

# build iterators
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter = BucketIterator.splits((train_ds, val_ds), batch_size=64,
                                             sort_within_batch=True, device=device)

# model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers, bidirectional, dropout, word_pad_idx):
        super(BiLSTM, self).__init__()
        self.emb = nn.Embedding(num_embeddings=input_dim,
                                embedding_dim=embedding_dim,
                                padding_idx=word_pad_idx)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                           num_layers=num_lstm_layers,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.emb(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output
