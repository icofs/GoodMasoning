import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import MorningClassifier
import pandas as pd
import re

# === Tokenization ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())[:20]

# === Vocabulary ===
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode(text, vocab, max_len=20):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# === Dataset ===
class TextDataset(Dataset):
    def __init__(self, df, vocab):
        self.vocab = vocab
        self.samples = [(encode(row["text"], vocab), row["label"]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float)

# === Load Data ===
df = pd.read_csv("data.csv")  # must have 'text' and 'label' columns
vocab = build_vocab(df["text"])
dataset = TextDataset(df, vocab)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# === Model Training ===
model = MorningClassifier(vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(10):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pt")
torch.save(vocab, "vocab.pt")
