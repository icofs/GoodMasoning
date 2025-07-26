import torch
import torch.nn as nn

class MorningClassifier(nn.Module):
    """
    Класс нейронной сети для классификации фраз на доброе утро/не доброе утро.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        pooled = embeds.mean(dim=1)                 # mean pooling
        x = self.relu(self.fc1(pooled))             # (batch, hidden_dim)
        x = self.sigmoid(self.fc2(x))               # (batch, 1)
        return x.squeeze(1)                         # (batch,)
