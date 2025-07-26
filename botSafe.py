import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

class MorningClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)         # (batch, seq_len, emb_dim)
        x = x.mean(dim=1)             # усредняем по seq_len, теперь (batch, emb_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(-1)


# === Токенизация ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())[:20]

# === Кодирование ===
def encode(text, vocab, max_len=20):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# === Загрузка словаря и модели ===
vocab = torch.load("vocab.pt")
model = MorningClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pt"))
model.eval()

# === Предсказание ===
def predict(text):
    input_ids = torch.tensor([encode(text, vocab)])
    with torch.no_grad():
        prob = model(input_ids).item()
    return prob

# === Обработка сообщений ===
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    confidence = predict(user_text)
    await update.message.reply_text(f"Уверенность в добром утре: {confidence:.3f}")

# === Команда /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "HW. Это пока только тест (в основном — датасета) для сего творения под названием \"Автоматическая базовая система упрощённого распознавания доброго утра\". Сокращённо: АБСУРДУ (я просто люблю хреновые аббревиатуры). Чтобы начать, просто отправьте сообщение. Медианная уверенность: 0.507, если больше — бот считает это добрым утром"
    )

# === Основная функция запуска бота ===
def main():
    BOT_TOKEN = "BOT_TOKEN"

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()

if __name__ == "__main__":
    main()
