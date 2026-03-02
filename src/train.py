import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")

    return acc, f1


if __name__ == "__main__":

    set_seed(42)

    wandb.init(
        project="imdb-sentiment",
        config={
            "model": "distilbert-base-uncased",
            "epochs": 3,
            "batch_size": 16,
            "lr": 2e-5,
            "max_length": 256,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = load_dataset("imdb")

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    writer = SummaryWriter("logs")

    best_f1 = 0

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("checkpoints", exist_ok=True)
            model.save_pretrained("checkpoints")
            tokenizer.save_pretrained("checkpoints")
            print("Best model saved.")

        try:
            wandb.log({
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "epoch": epoch + 1
            })
        except Exception as e:
            print(f"Warning: WandB logging failed — {e}. Continuing training.")

    writer.close()
    try:
        wandb.finish()
    except Exception:
        pass