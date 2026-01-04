import torch
from dataset import LongDocDataset
from collate import collate_longdoc
from model import LongDocClassifier
from config import LongDocConfig


config = LongDocConfig()
model = LongDocClassifier(config, task_ids=[0, 1])
model.eval()


sample = LongDocDataset(["finance risk document" * 100], [1], [0])[0]
batch = collate_longdoc([sample])


with torch.no_grad():
    out = model(batch)
    print(torch.sigmoid(out["logits"]))
