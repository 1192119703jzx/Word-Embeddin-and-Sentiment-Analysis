import torch
import torch.nn as nn

class FFN_enhanced(nn.Module):
    def __init__(self, embed_dim, num_hidden, labels_num):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, num_hidden, bias=True)
        self.linear_2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.linear_3 = nn.Linear(num_hidden, labels_num, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.linear_1(x)
        h1 = self.relu(h1)
        h2 = self.linear_2(h1)
        h2 = self.relu(h2)
        o = self.linear_3(h2)
        return o

class LSTMEncoder_enhanced(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_classes, num_layers, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2, num_classes)
        else:
            self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (batch_size, hidden_dim)
        lstm_hidden, _ = self.lstm(x)
        lstm_hidden = lstm_hidden.unsqueeze(1)

        # Use the hidden state of the last LSTM cell for classification
        if self.lstm.bidirectional:
            lstm_hidden = torch.cat((lstm_hidden[:, -1, :self.lstm.hidden_size], lstm_hidden[:, 0, self.lstm.hidden_size:]), dim=-1)
        else:
            lstm_hidden = lstm_hidden[:, -1, :]

        # (batch_size, num_classes)
        logits = self.linear(lstm_hidden)
        return logits