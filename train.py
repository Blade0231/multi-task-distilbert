import torch
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from dataset import LongDocDataset
from collate import collate_longdoc
from model import LongDocClassifier
from config import LongDocConfig
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


def build_dataset(df: pd.DataFrame):
    train = pd.DataFrame()
    test = pd.DataFrame()
    val = pd.DataFrame()

    for i in [1, 2, 3]:
        df_ = df[["Text", i]].head(20)
        df_.columns = ["Text", "Label"]
        df_["TaskId"] = i
        train_, test_ = train_test_split(df_, test_size=0.2, stratify=df_["Label"])
        train_, val_ = train_test_split(train_, test_size=0.2, stratify=train_["Label"])
        train = pd.concat([train_, train])
        test = pd.concat([test_, test])
        val = pd.concat([val_, val])

    return train.dropna(), test.dropna(), val.dropna()


def main():
    df = pd.read_excel("data/Dataset.xlsx")

    train, test, val = build_dataset(df)

    train_texts, train_labels, train_task_ids = (
        train["Text"].tolist(),
        train["Label"].tolist(),
        train["TaskId"].tolist(),
    )
    test_texts, test_labels, test_task_ids = (
        test["Text"].tolist(),
        test["Label"].tolist(),
        test["TaskId"].tolist(),
    )
    val_texts, val_labels, val_task_ids = (
        val["Text"].tolist(),
        val["Label"].tolist(),
        val["TaskId"].tolist(),
    )

    texts = train["Text"].tolist()
    labels = train["Label"].tolist()
    task_ids = train["TaskId"].tolist()

    config = LongDocConfig()

    train_dataset = LongDocDataset(train_texts, train_labels, train_task_ids)
    test_dataset = LongDocDataset(test_texts, test_labels, test_task_ids)
    val_dataset = LongDocDataset(val_texts, val_labels, val_task_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_longdoc)
    test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_longdoc)
    val_dataloader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_longdoc)

    mlflow.set_experiment("longdoc_multitask_classifier")

    model = LongDocClassifier(config, task_ids=[1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with mlflow.start_run():
        mlflow.log_params(config.__dict__)

    model.train()

    for step, batch in enumerate(train_dataloader):
        out = model(batch)
        logits = out["logits"].detach().cpu()
        labels = out["labels"].detach().cpu()
        tasks = batch["task_ids"].detach().cpu()

        eval_loss = loss_fn(logits, labels)

        probs = torch.sigmoid(logits)

        task_preds = {}
        task_labels = {}

        for p, y, t in zip(probs, labels, tasks):
            t = int(t.item())
            task_preds.setdefault(t, []).append(p.item())
            task_labels.setdefault(t, []).append(y.item())

        loss = loss_fn(out["logits"], out["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Evaluation (Task-wise) ----
        model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                out = model(batch)
                eval_loss = loss_fn(out["logits"], out["labels"])
                eval_losses.append(eval_loss.item())

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        mlflow.log_metric("val_loss", avg_eval_loss)

        # ---- Task-wise metrics ----
        for task_id in task_preds:
            y_true = task_labels[task_id]
            y_prob = task_preds[task_id]
            y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float("nan")

            mlflow.log_metric(f"val_acc_task_{task_id}", acc)
            mlflow.log_metric(f"val_auc_task_{task_id}", auc)
        mlflow.log_metric("val_loss", avg_eval_loss)

        # mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_metric("train_loss", loss.item(), step=step)

        


if __name__ == "__main__":
    main()
