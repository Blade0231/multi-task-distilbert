import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from typing import List


def chunk_text(text, tokenizer, max_len=512, stride=64):
    enc = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_len,
        stride=stride,
        add_special_tokens=True,
    )

    return enc["input_ids"], enc["attention_mask"]


class LongDocDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], task_ids: List[int]):
        self.texts = texts
        self.labels = labels
        self.task_ids = task_ids
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_masks = chunk_text(self.texts[idx], self.tokenizer)

        return {
            "input_ids": [torch.tensor(x) for x in input_ids],
            "attention_mask": [torch.tensor(x) for x in attention_masks],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "task_id": self.task_ids[idx],
        }
