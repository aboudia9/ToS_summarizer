import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import id2label, BERT_MAX_LENGTH


# Class definition to reference later for reusability and such
class TOSDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=BERT_MAX_LENGTH):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Returns total number of samples — PyTorch needs this
        return len(self.data)

    def __getitem__(self, index):
        # Grab the sentence and label at the given index
        sentence = str(self.data.iloc[index]['sentence'])
        label = int(self.data.iloc[index]['label'])

        # Tokenize the sentence
        # padding/truncation ensures all inputs are the same length
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_bert(model_path, device):
    # Saved weights may be fp16 (compressed for the app) — always load as fp32,
    # since half precision is slow/unsupported for a lot of ops on CPU
    bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float32)
    bert_model = bert_model.to(device)
    bert_model.eval()
    return bert_model, bert_tokenizer


# Run inference on a dataframe and add predicted labels (used by the training scripts)
def run_inference(df, model, tokenizer, device, batch_size=32):
    df['bert_label'] = [
        id2label[pred] for pred in classify_clauses(
            df['sentence'].astype(str).tolist(), model, tokenizer, device, batch_size
        ).argmax(dim=1).tolist()
    ]
    return df


def classify_clauses(clauses, model, tokenizer, device, batch_size=32):
    # Batched version of the old one-clause-at-a-time loop. Padding to the longest
    # clause in each batch (instead of always 256) gives the same predictions,
    # just a lot faster. Batches group clauses of similar length so there's even less
    # padding. Returns a (num_clauses, 3) tensor of class probabilities, in input order
    model.eval()
    order = sorted(range(len(clauses)), key=lambda i: len(clauses[i]))
    all_probs = []

    for i in range(0, len(order), batch_size):
        inputs = tokenizer(
            [clauses[j] for j in order[i:i + batch_size]],
            max_length=BERT_MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            all_probs.append(torch.softmax(outputs.logits, dim=1).cpu())

    if not all_probs:
        return torch.empty((0, len(id2label)))
    probs = torch.empty((len(clauses), len(id2label)))
    probs[order] = torch.cat(all_probs)
    return probs
