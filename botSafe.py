import re
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from model import MorningClassifier
from utils import tokenize, encode, load_vocab

# === Параметры ===
MODEL_PATH = "model.pt"
VOCAB_PATH = "vocab.pt"
MAX_LEN = 20

# === Загрузка словаря и модели ===
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

# === Обработка сообщений ===
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений."""
    user_text = update.message.text
    confidence = predict(user_text)
    await update.message.reply_text(f"Уверенность в добром утре: {confidence:.3f}")

# === Команда /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "HW. Это пока только тест (в основном — датасета) для сего творения под названием \"Автоматическая базовая система упрощённого распознавания доброго утра\". Сокращённо: АБСУРДУ (я просто люблю хреновые аббревиатуры). Чтобы начать, просто отправьте сообщение. Медианная уверенность: 0.507, если больше — бот считает это добрым утром"
    )

# === Основная функция запуска бота ===
def main():
    """Запуск Telegram-бота."""
    BOT_TOKEN = "BOT_TOKEN"
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()

if __name__ == "__main__":
    main()
