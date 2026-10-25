---
title: Clause & Effect
emoji: 📜
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8501
pinned: false
short_description: Flags unfair clauses in Terms of Service agreements
---

# Clause & Effect — ToS Summarizer

Paste a Terms of Service agreement (or upload a `.txt`) and get back the clauses worth
worrying about, rewritten in plain English.

1. The document is split into clauses (NLTK sentence tokenization, headings filtered out).
2. A fine-tuned [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) classifies each
   clause as clearly fair, potentially unfair, or clearly unfair.
3. A fine-tuned [BART](https://huggingface.co/facebook/bart-large-cnn) summarizes the flagged ones.

Models: [bigboub/clause-and-effect-models](https://huggingface.co/bigboub/clause-and-effect-models)

Runs on a free CPU Space — a full-length ToS takes a minute or two, and the first visit after the
Space has been idle may take a bit longer while it wakes up.

*Not legal advice.*
