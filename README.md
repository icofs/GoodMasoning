# GoodMasoning

Классификация фраз на "доброе утро"/"не доброе утро" с помощью PyTorch и Telegram-бота.

## Структура
- `model.py` — определение нейросети
- `utils.py` — токенизация, кодирование, работа со словарём
- `train.py` — обучение модели
- `predict.py` — консольное предсказание
- `botSafe.py` — Telegram-бот
- `data.csv` — датасет (столбцы: text, label)
- `model.pt`, `vocab.pt` — веса и словарь

## Установка
```bash
pip install -r requirements.txt
```

## Обучение
```bash
python train.py
```

## Предсказание в консоли
```bash
python predict.py
```

## Запуск Telegram-бота
1. Вставьте свой токен в переменную `BOT_TOKEN` в `botSafe.py`.
2. Запустите:
```bash
python botSafe.py
``` 