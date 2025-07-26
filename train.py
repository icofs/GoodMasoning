import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import MorningClassifier
from utils import tokenize, encode, build_vocab, load_vocab, save_vocab
import pandas as pd

# === Параметры ===
DATA_PATH = "data.csv"
MODEL_PATH = "model.pt"
VOCAB_PATH = "vocab.pt"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
MAX_LEN = 20

# === Dataset ===
class TextDataset(Dataset):
    def __init__(self, df, vocab):
        self.vocab = vocab
        self.samples = [(encode(row["text"], vocab, max_len=MAX_LEN), row["label"]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float)

# === Загрузка данных ===
def load_data(path):
    """Загрузить датасет из CSV."""
    return pd.read_csv(path)

df = load_data(DATA_PATH)  # должен содержать 'text' и 'label'
vocab = build_vocab(df["text"])
save_vocab(vocab, VOCAB_PATH)
dataset = TextDataset(df, vocab)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Обучение модели ===
def train():
    """Обучить модель и сохранить лучшую по loss."""
    model = MorningClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"Best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
