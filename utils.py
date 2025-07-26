import re
import torch


def tokenize(text):
    """Разбить текст на токены (слова), привести к нижнему регистру, ограничить длину 20."""
    return re.findall(r"\b\w+\b", text.lower())[:20]


def encode(text, vocab, max_len=20):
    """Преобразовать текст в список индексов по словарю."""
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def build_vocab(texts):
    """Построить словарь из списка текстов."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def load_vocab(path):
    """Загрузить словарь из файла."""
    return torch.load(path)


def save_vocab(vocab, path):
    """Сохранить словарь в файл."""
    torch.save(vocab, path) 