"""Stage 3 — fine-tune BART on the distilled plain-English clause summaries.
(From notebooks/ToS_bart_summarization.ipynb)

Expects df_train.csv / df_val.csv plus the distilled df_*_bart_distilled.csv files in DATA_DIR,
and a trained BERT model (used to label the EE21 supplemental dataset).

    python -m training.train_bart --out bart_model_v3
"""
import argparse

import pandas as pd
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup

from tos_summarizer.config import ROOT, DATA_DIR, BART_BASE_MODEL, label2id, get_device
from tos_summarizer.classifier import load_bert, run_inference
from tos_summarizer.summarizer import TOSSummarizationDataset

REPLACEMENT = "the applicable jurisdiction"

# Telltale phrases that indicate a refusal/meta-commentary got saved as a "summary"
# instead of an actual plain-English rewrite
refusal_markers = [
    "I can see this",
    "I don't see",
    "I'd be happy to help",
    "Could you please provide",
    "no substantive",
    "section heading rather than",
    "does not contain",
    "appears to be a",
    "please provide the",
]


def looks_like_refusal(text):
    text = str(text).lower()
    return any(marker.lower() in text for marker in refusal_markers)


def train_bart_epoch(model, dataloader, optimizer, scheduler, device):
    # Set model to training mode
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass — BART computes loss internally when labels are provided
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights and scheduler
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def eval_bart_epoch(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "bart_model_v3"))
    parser.add_argument("--bert", default=str(ROOT / "bert_model"),
                        help="Trained BERT used to label the EE21 supplemental dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)  # BART is much bigger than BERT
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop-refusals", action="store_true",
                        help="Drop distilled rows whose summary looks like a refusal/meta-comment")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load the supplemental summarization dataset and fix the "location X" placeholder
    df_summaries = pd.DataFrame(load_dataset("EE21/ToS-Summaries")['train'])
    df_summaries["summary"] = df_summaries["summary"].str.replace("location X", REPLACEMENT, regex=False)
    print(f"Remaining 'location X' occurrences: {df_summaries['summary'].str.contains('location X').sum()}")

    # Rename plain_text to sentence so it matches our existing DataFrames
    df_summaries = df_summaries.rename(columns={'plain_text': 'sentence'})

    # Run BERT inference on the summaries dataset to get unfairness labels
    print("Running BERT inference on summaries dataset...")
    bert_model, bert_tokenizer = load_bert(args.bert, device)
    df_summaries = run_inference(df_summaries, bert_model, bert_tokenizer, device)
    print(df_summaries['bert_label'].value_counts())
    del bert_model

    # Load the real distilled summaries generated via the batch job earlier
    df_train_for_bart = pd.read_csv(DATA_DIR / 'df_train_bart_distilled.csv')
    df_val_for_bart = pd.read_csv(DATA_DIR / 'df_val_bart_distilled.csv')

    train_bad = df_train_for_bart['summary'].apply(looks_like_refusal)
    val_bad = df_val_for_bart['summary'].apply(looks_like_refusal)
    print(f"Distilled rows that look like refusals — train: {train_bad.sum()} | val: {val_bad.sum()}")
    if args.drop_refusals:
        df_train_for_bart = df_train_for_bart[~train_bad]
        df_val_for_bart = df_val_for_bart[~val_bad]

    # Align EE21 supplemental data to the same column structure
    df_summaries['unfairness_level'] = df_summaries['bert_label']
    df_summaries['label'] = df_summaries['bert_label'].map(label2id)

    # Combine: real distilled TOSDatasetV3 summaries + real EE21 summaries
    df_train_final = pd.concat(
        [df_train_for_bart, df_summaries[['sentence', 'unfairness_level', 'label', 'summary']]],
        ignore_index=True
    )
    df_val_final = df_val_for_bart.copy()
    print(f"Train rows: {len(df_train_final)} | Val rows: {len(df_val_final)}")

    tokenizer = BartTokenizer.from_pretrained(BART_BASE_MODEL)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(TOSSummarizationDataset(df_train_final, tokenizer),
                              batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(TOSSummarizationDataset(df_val_final, tokenizer),
                            batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    # Load the BART model pretrained on CNN/DailyMail news summarization
    model = BartForConditionalGeneration.from_pretrained(BART_BASE_MODEL).to(device)

    # Same AdamW as BERT but lower learning rate — BART is bigger and more sensitive
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        train_loss = train_bart_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = eval_bart_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # Save model checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"✅ Model improved and saved to {args.out}")
        else:
            print(f"⚠️ No improvement this epoch")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
