---
language: en
tags:
  - legal
  - terms-of-service
  - text-classification
  - summarization
datasets:
  - CodeHima/TOS_DatasetV3
  - EE21/ToS-Summaries
---

# Clause & Effect models

The two fine-tuned models behind the [Clause & Effect Space](https://huggingface.co/spaces/bigboub/clause-and-effect).
Weights are stored in fp16 to halve the download size; load them as fp32 (`dtype=torch.float32`) for CPU inference.

| Folder | Base model | Task |
|---|---|---|
| `bert/` | `nlpaueb/legal-bert-base-uncased` | 3-way clause fairness: `clearly_fair`, `potentially_unfair`, `clearly_unfair` |
| `bart/` | `facebook/bart-large-cnn` | Plain-English summary of a clause, input formatted as `"{label}: {clause}"` |

## BERT test-set results (CodeHima/TOS_DatasetV3, 960 clauses)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| clearly_fair | 0.92 | 0.93 | 0.92 |
| potentially_unfair | 0.81 | 0.80 | 0.81 |
| clearly_unfair | 0.79 | 0.80 | 0.80 |
| **Accuracy** | | | **0.864** |

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration

repo = "bigboub/clause-and-effect-models"
bert_tok = AutoTokenizer.from_pretrained(repo, subfolder="bert")
bert = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="bert", dtype=torch.float32)
bart_tok = AutoTokenizer.from_pretrained(repo, subfolder="bart")
bart = BartForConditionalGeneration.from_pretrained(repo, subfolder="bart", dtype=torch.float32)
```

## Limitations

Trained on individual, clean ToS sentences. On full real-world documents the classifier
over-flags some boilerplate (e.g. "These Terms are an agreement between you and ...") as unfair.
Not legal advice.
