import torch
import torch.nn as nn
from dataloader import BuildDataset

class LSTMNet(nn.Module):

    def __init__(self,vocab_size=25000, embedding_dim=200, input_dim=None, hidden_dim=256, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.2, pad=None):
        super().__init__()
        self.build_data = BuildDataset()
        self.vocalb_size = vocab_size
        self.input_dim = input_dim #
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        # pad sentences
        self.pad_idx = pad

        self.embedding_layer = nn.Embedding(self.vocalb_size, self.embedding_dim, padding_idx=self.pad_idx)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, self.bidirectional, self.dropout)
        self.fc1 = nn.Linear(self.hidden_dim*2 , self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(float(self.dropout))

    def forward(self, text, text_lengths):
        embedded = self.embedding_layer(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc_out(output))
        return output




