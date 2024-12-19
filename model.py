import torch
import torch.nn as nn

class FFN_static(nn.Module):
    def __init__(self, embed_dim, num_hidden, labels_num, embedding_matrix):
        super(FFN_static, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=True)

        self.linear_1 = nn.Linear(embed_dim, num_hidden, bias=True)
        self.linear_2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.linear_3 = nn.Linear(num_hidden, labels_num, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.mean(dim=1)

        h1 = self.linear_1(embeds)
        h1 = self.relu(h1)
        h2 = self.linear_2(h1)
        h2 = self.relu(h2)
        o = self.linear_3(h2)
        return o

class LSTMEncoder_static(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_classes, embedding_matrix, num_layers, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2, num_classes)
        else:
            self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        # (batch_size, seq_len, embed_dim)
        input_embeds = self.embedding(input_ids)

        # (batch_size, seq_len, hidden_dim)
        lstm_hidden, _ = self.lstm(input_embeds)

        # Use the hidden state of the last LSTM cell for classification
        if self.lstm.bidirectional:
            lstm_hidden = torch.cat((lstm_hidden[:, -1, :self.lstm.hidden_size], lstm_hidden[:, 0, self.lstm.hidden_size:]), dim=-1)
        else:
            lstm_hidden = lstm_hidden[:, -1, :]

        # (batch_size, num_classes)
        logits = self.linear(lstm_hidden)
        return logits