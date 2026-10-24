"""Stage 2 — fine-tune Legal-BERT to classify clause fairness.
(From notebooks/ToS_bert_embeddings.ipynb)

    python -m training.train_bert --out bert_model_v2
    python -m training.train_bert --out bert_model_v2 --select-on f1 --epochs 5

Then point the app at it: BERT_MODEL_PATH=bert_model_v2, or re-run
scripts/export_models.py --bert bert_model_v2 to compress it into models/bert.
"""
import argparse

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from tos_summarizer.config import ROOT, DATA_DIR, BERT_BASE_MODEL, id2label, label2id, get_device
from tos_summarizer.classifier import TOSDataset


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    # Set model to training mode for dropout and gradient updates
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in dataloader:
        # Move batch tensors to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Zero out gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Calculate loss using our weighted loss function
        loss = loss_fn(logits, labels)

        # Track correct predictions for accuracy
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        # Backward pass — compute gradients
        loss.backward()

        # Clip gradients to prevent exploding gradient problem
        # common practice for transformer fine-tuning (AI suggestion here)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights and learning rate
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy


def eval_epoch(model, dataloader, loss_fn, device):
    # Set model to evaluation mode — disables dropout
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    # Disable gradient computation for evaluation — saves on memory and speeds things up
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            all_preds += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, macro_f1, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "bert_model_v2"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)  # reduce to 8 if there are memory issues
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--select-on", choices=["loss", "f1"], default="loss",
                        help="Which validation metric decides the saved checkpoint")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load preprocessed splits from CSV
    df_train = pd.read_csv(DATA_DIR / 'df_train.csv')
    df_val = pd.read_csv(DATA_DIR / 'df_val.csv')
    df_test = pd.read_csv(DATA_DIR / 'df_test.csv')
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(TOSDataset(df_train, tokenizer), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(TOSDataset(df_val, tokenizer), batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(TOSDataset(df_test, tokenizer), batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    # We need to explicitly define label mappings so the model knows our 3 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True   # bypasses the label mismatch warning
    ).to(device)

    # Calculate class weights to handle our slight data imbalance
    # The idea is that rarer classes get higher weight so the model doesn't ignore them
    class_counts = df_train['label'].value_counts().sort_index().values
    class_weights = torch.tensor(
        1.0 / class_counts / (1.0 / class_counts).sum(),
        dtype=torch.float
    ).to(device)

    # Loss function — CrossEntropy with our class weights baked in
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler gradually reduces learning rate over training
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # warm up for the first 10% of steps
        num_training_steps=total_steps
    )

    # Track the best validation score so we only save improved models
    best = float('inf') if args.select_on == "loss" else -1.0

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc, val_f1, _ = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        # Save model checkpoint if validation score improved
        improved = val_loss < best if args.select_on == "loss" else val_f1 > best
        if improved:
            best = val_loss if args.select_on == "loss" else val_f1
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"✅ Model improved and saved to {args.out}")
        else:
            print(f"⚠️ No improvement this epoch")

    # Final report on the held-out test split using the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    _, test_acc, test_f1, test_preds = eval_epoch(model, test_loader, loss_fn, device)
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
    print(classification_report(df_test['label'], test_preds, target_names=list(label2id), digits=3))


if __name__ == "__main__":
    main()
