import torch
from model import MorningClassifier
from utils import encode, load_vocab

# === Параметры ===
MODEL_PATH = "model.pt"
VOCAB_PATH = "vocab.pt"
MAX_LEN = 20

# === Загрузка модели и словаря ===
vocab = load_vocab(VOCAB_PATH)
model = MorningClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Предсказание ===
def predict(text):
    """Вернуть вероятность того, что текст — доброе утро."""
    input_ids = torch.tensor([encode(text, vocab, max_len=MAX_LEN)])
    with torch.no_grad():
        prob = model(input_ids).item()
    return prob

# === Пример ===
while True:
    text = input("Enter a phrase: ")
    if not text:
        break
    confidence = predict(text)
    print(f"Confidence: {confidence:.3f}")
