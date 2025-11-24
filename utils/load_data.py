import torch
from datasets import load_dataset


def load_data():
    cache_dir = r"D:\Project\move-review-sentiment-analysis-model\aclImdb"

    train_ds = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir, split="train")
    test_ds = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir, split="test")

    return train_ds, test_ds
