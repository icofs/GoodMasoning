import torch
from model import MorningClassifier
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())[:20]

def encode(text, vocab, max_len=20):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# === Load model & vocab ===
vocab = torch.load("vocab.pt")
model = MorningClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pt"))
model.eval()

# === Predict ===
def predict(text):
    input_ids = torch.tensor([encode(text, vocab)])
    with torch.no_grad():
        prob = model(input_ids).item()
    return prob

# === Example ===
while True:
    text = input("Enter a phrase: ")
    if not text:
        break
    confidence = predict(text)
    print(f"Confidence: {confidence:.3f}")
