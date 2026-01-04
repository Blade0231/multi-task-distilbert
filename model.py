import torch
import torch.nn as nn
from transformers import DistilBertModel
from peft import LoraConfig, get_peft_model, TaskType
from config import LongDocConfig


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x):
        scores = self.scorer(x).squeeze(-1)
        scores = scores - scores.max()
        weights = torch.softmax(scores, dim=0)
        return torch.sum(weights.unsqueeze(-1) * x, dim=0)


class LongDocClassifier(nn.Module):
    def __init__(self, config: LongDocConfig, task_ids):
        super().__init__()
        base_encoder = DistilBertModel.from_pretrained(config.base_model_name)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_lin", "k_lin", "v_lin", "ffn_lin"],
        )

        self.encoder = get_peft_model(base_encoder, lora_config)
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_dim, config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_dim, config.shared_dim // 2),
            nn.GELU(),
        )
        self.pooler = AttentionPooling(config.shared_dim // 2)
        self.task_heads = nn.ModuleDict(
            {str(t): nn.Linear(config.shared_dim // 2, 1) for t in task_ids}
        )

    def forward(self, batch):
        device = next(self.parameters()).device

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True
        ).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch["attention_mask"], batch_first=True
        ).to(device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embs = outputs.last_hidden_state[:, 0, :]
        cls_embs = self.shared(cls_embs)

        docs, offset = [], 0
        for n in batch["chunk_counts"]:
            docs.append(self.pooler(cls_embs[offset : offset + n]))
            offset += n

        docs = torch.stack(docs)
        logits = []
        for i, tid in enumerate(batch["task_ids"]):
            logits.append(self.task_heads[str(int(tid))](docs[i]))

        return {
            "logits": torch.stack(logits).squeeze(-1),
            "labels": batch["labels"].to(device),
        }
