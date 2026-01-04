import torch


def collate_longdoc(batch):
    input_ids, attention_masks, chunk_counts = [], [], []
    labels, task_ids = [], []

    for sample in batch:
        n = len(sample["input_ids"])
        chunk_counts.append(n)
        input_ids.extend(sample["input_ids"])
        attention_masks.extend(sample["attention_mask"])
        labels.append(sample["label"])
        task_ids.append(sample["task_id"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "chunk_counts": chunk_counts,
        "labels": torch.stack(labels),
        "task_ids": torch.tensor(task_ids),
    }
