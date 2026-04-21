# ToS_summarizer  

Terms of Service (ToS) summarizer to provide the user with the important info from long ToS documents. It will flag unfair clauses and show the user any potentially concerning or compromising clauses within the document making dense legal language more accessible to everyday users and allowing the user to make more informed decisions while navigating the web.  

This project was built as a capstone for CS561. It consists of three main stages: preprocessing, BERT-based clause classification, and BART-based abstractive summarization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Local Setup](#local-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline Locally](#running-the-pipeline-locally)
- [Google Colab Setup](#google-colab-setup)
  - [Environment Configuration](#environment-configuration)
  - [Running the Pipeline on Colab](#running-the-pipeline-on-colab)
- [Output](#output)
- [Notes](#notes)

---

## Project Overview

The pipeline works in three sequential stages:

1. **Preprocessing** — Loads and cleans the raw ToS dataset, splits it into train/validation/test sets, and prepares clause-level records for classification.
2. **BERT Classification** — Fine-tunes `nlpaueb/legal-bert-base-uncased` to classify each clause as `clearly_fair`, `potentially_unfair`, or `clearly_unfair`. Labeled CSVs are saved after inference.
3. **BART Summarization** — Fine-tunes `facebook/bart-large-cnn` on the labeled clause data (supplemented with a plain-English summarization dataset) to generate human-readable summaries for each ToS agreement.

---

## Repository Structure

```
ToS_summarizer/
├── preprocessing/
│   └── preprocessing.ipynb        # Data cleaning and split logic
├── classification/
│   └── bert_classification.ipynb  # BERT fine-tuning and inference
├── summarization/
│   └── bart_summarization.ipynb   # BART fine-tuning and summary generation
├── data/
│   └── (labeled CSVs saved here after BERT inference)
└── README.md
```

> **Note:** Model weights are not committed to the repository due to file size. See the [Output](#output) section for where models are saved locally and on Drive.

---

## Dataset

The project uses [`CodeHima/TOSDatasetV3`](https://huggingface.co/datasets/CodeHima/TOSDatasetV3) from HuggingFace, which includes pre-existing train, validation, and test splits. These splits are used as-is — do not re-split manually, as doing so discards validated data boundaries.

---

## Models

| Stage | Model | Source |
|---|---|---|
| Classification | `nlpaueb/legal-bert-base-uncased` | HuggingFace |
| Summarization | `facebook/bart-large-cnn` | HuggingFace |

---

## Local Setup

### Prerequisites

- Python 3.9+
- PyCharm (recommended) with Jupyter notebook integration, or JupyterLab
- A CUDA-compatible GPU is strongly recommended for fine-tuning. CPU execution is possible but impractically slow for transformer training.

### Installation

**Step 1.** Clone the repository:

```bash
git clone git@github.com:aboudia9/ToS_summarizer.git
cd ToS_summarizer
```

**Step 2.** Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**Step 3.** Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets pandas scikit-learn
pip install jupyter notebook
```

> If you have a CUDA GPU, make sure you install the CUDA-compatible version of PyTorch. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and use the selector to get the right install command for your system.

### Running the Pipeline Locally

Run the notebooks **in order**. Each notebook depends on the outputs of the one before it.

---

#### Step 1 — Preprocessing (`preprocessing/preprocessing.ipynb`)

This notebook loads the raw dataset from HuggingFace and prepares it for classification.

1. Open `preprocessing/preprocessing.ipynb` in PyCharm or Jupyter.
2. Run all cells in order from top to bottom.
3. The notebook will:
   - Load all three HuggingFace dataset splits (train, validation, test)
   - Apply `clean_df()` to normalize text
   - Save cleaned DataFrames locally (adjust the output path in the notebook if needed)

---

#### Step 2 — BERT Classification (`classification/bert_classification.ipynb`)

This notebook fine-tunes Legal-BERT on the cleaned clause data and runs inference to produce labeled CSVs.

1. Open `classification/bert_classification.ipynb`.
2. Update the file paths at the top of the notebook to point to the CSVs saved in Step 1.
3. Run all cells in order. Key things that happen:
   - Tokenizes clauses using the `nlpaueb/legal-bert-base-uncased` tokenizer
   - Fine-tunes for 3 epochs
   - Runs inference on train/val/test splits
   - Saves `df_train_labeled.csv`, `df_val_labeled.csv`, and `df_test_labeled.csv` to the `data/` directory
4. After training, the model weights are saved locally. Update the `model_save_path` variable in the notebook to set your preferred save location.

> Validation accuracy after fine-tuning should be approximately **84.55%** across 3 epochs. If it's significantly lower, double-check that the dataset splits weren't re-split manually.

---

#### Step 3 — BART Summarization (`summarization/bart_summarization.ipynb`)

This notebook fine-tunes BART on the labeled clause data to generate plain-English summaries.

1. Open `summarization/bart_summarization.ipynb`.
2. Update the file paths to point to the labeled CSVs from Step 2.
3. Make sure a supplemental summarization dataset with plain-English summary targets is available and its path is configured at the top of the notebook.
4. Run all cells in order.

---

## Google Colab Setup

Colab is the recommended environment for GPU-accelerated training. The project uses Google Drive for persistent storage of model checkpoints and labeled data.

### Environment Configuration

#### Step 1 — Set the Runtime to GPU

Before running anything:

1. In Colab, go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 is sufficient)
3. Click **Save**

To verify the GPU is active, run this in a cell:

```python
import torch
print(torch.cuda.is_available())        # Should print: True
print(torch.cuda.get_device_name(0))    # Should print the GPU model, e.g., Tesla T4
```

#### Step 2 — Mount Google Drive

Run this at the top of each notebook before any file I/O:

```python
from google.colab import drive
drive.mount('/content/drive')
```

You'll be prompted to authorize access. After mounting, your Drive is accessible at `/content/drive/MyDrive/`.

The recommended project directory on Drive is:

```
/content/drive/MyDrive/ToS_summarizer/
```

#### Step 3 — Install Dependencies

Colab has most packages pre-installed, but run this at the top of each notebook to make sure everything is available:

```python
!pip install -q transformers datasets
```

> `torch`, `pandas`, and `scikit-learn` are already available in the default Colab environment.

#### Step 4 — Clone the Repository (Optional)

If you want to pull the latest code from GitHub directly into Colab:

```python
!git clone git@github.com:aboudia9/ToS_summarizer.git /content/ToS_summarizer
```

> SSH may require additional key setup in Colab. If you run into auth issues, use HTTPS:
> ```python
> !git clone https://github.com/aboudia9/ToS_summarizer.git /content/ToS_summarizer
> ```

---

### Running the Pipeline on Colab

#### Step 1 — Preprocessing

1. Open `preprocessing/preprocessing.ipynb` in Colab (via File → Upload or from Drive).
2. Make sure Drive is mounted (see Step 2 above).
3. Set output paths to save to Drive:

```python
# Example output paths
TRAIN_OUTPUT = "/content/drive/MyDrive/ToS_summarizer/data/df_train_cleaned.csv"
VAL_OUTPUT   = "/content/drive/MyDrive/ToS_summarizer/data/df_val_cleaned.csv"
TEST_OUTPUT  = "/content/drive/MyDrive/ToS_summarizer/data/df_test_cleaned.csv"
```

4. Run all cells in order. Cleaned CSVs will be saved to Drive so they persist after the Colab session ends.

---

#### Step 2 — BERT Classification

1. Open `classification/bert_classification.ipynb` in Colab.
2. Mount Drive and set input/output paths:

```python
# Input — cleaned CSVs from preprocessing
TRAIN_CSV = "/content/drive/MyDrive/ToS_summarizer/data/df_train_cleaned.csv"
VAL_CSV   = "/content/drive/MyDrive/ToS_summarizer/data/df_val_cleaned.csv"
TEST_CSV  = "/content/drive/MyDrive/ToS_summarizer/data/df_test_cleaned.csv"

# Output — labeled CSVs after BERT inference
TRAIN_LABELED = "/content/drive/MyDrive/ToS_summarizer/data/df_train_labeled.csv"
VAL_LABELED   = "/content/drive/MyDrive/ToS_summarizer/data/df_val_labeled.csv"
TEST_LABELED  = "/content/drive/MyDrive/ToS_summarizer/data/df_test_labeled.csv"

# Model checkpoint save path
MODEL_SAVE_PATH = "/content/drive/MyDrive/ToS_summarizer/bert_model"
```

3. Run all cells in order. Training runs for 3 epochs on the T4 GPU. The model and labeled CSVs will be saved to Drive.

> **Colab timeout warning:** Colab free tier sessions can disconnect after ~90 minutes of inactivity. Make sure the model is saving to Drive so you don't lose progress. If the session disconnects mid-training, remount Drive and reload from the last saved checkpoint.

---

#### Step 3 — BART Summarization

1. Open `summarization/bart_summarization.ipynb` in Colab.
2. Mount Drive and set paths:

```python
# Input — labeled clause CSVs from BERT step
TRAIN_LABELED = "/content/drive/MyDrive/ToS_summarizer/data/df_train_labeled.csv"
VAL_LABELED   = "/content/drive/MyDrive/ToS_summarizer/data/df_val_labeled.csv"

# Path to supplemental summarization dataset (with plain-English summary targets)
SUMM_DATASET_PATH = "/content/drive/MyDrive/ToS_summarizer/data/supplemental_summary_data.csv"

# BART model save path
BART_SAVE_PATH = "/content/drive/MyDrive/ToS_summarizer/bart_model"
```

3. Make sure the GPU runtime is active before starting training (this step is significantly more compute-heavy than BERT).
4. Run all cells in order.

---

## Output

After all three stages complete, the pipeline produces:

- `df_train_labeled.csv`, `df_val_labeled.csv`, `df_test_labeled.csv` — clause-level data with fairness labels from BERT
- Fine-tuned BERT model weights saved to `bert_model/`
- Fine-tuned BART model weights saved to `bart_model/`
- Plain-English summaries generated for each ToS document in the test set

---

## Notes

- The HuggingFace dataset comes with pre-existing train/val/test splits. These are intentional and should not be overridden with a manual `train_test_split()` call. Doing so discards the validated split boundaries and can introduce data leakage.
- Local CPU training is not recommended for either the BERT or BART steps. Even fine-tuning for a few epochs takes impractically long without a GPU.
- SSH is the more reliable authentication method for GitHub when working inside Colab. If you're running into push/pull issues, switching from HTTPS to SSH typically resolves them.
- For any issues with GPU availability mid-session on Colab, go to **Runtime → Disconnect and delete runtime**, then reconnect and re-verify GPU access before resuming.
