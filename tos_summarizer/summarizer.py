import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

from .config import BART_MAX_INPUT_LENGTH, BART_MAX_TARGET_LENGTH


class TOSSummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length=BART_MAX_INPUT_LENGTH,
                 max_target_length=BART_MAX_TARGET_LENGTH):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Label now actually gets prepended to the input (this was a
        # comment-only claim before — never really happened)
        input_text = f"{row['unfairness_level']}: {row['sentence']}"

        # Real plain-English target, not a label+sentence echo
        target_text = str(row['summary'])

        input_encoding = self.tokenizer(
            input_text, max_length=self.max_input_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_target_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def load_bart(model_path, device):
    # Same as load_bert — stored weights may be fp16, but we run in fp32
    bart_model = BartForConditionalGeneration.from_pretrained(model_path, dtype=torch.float32)
    bart_tokenizer = BartTokenizer.from_pretrained(model_path)
    bart_model = bart_model.to(device)
    bart_model.eval()
    return bart_model, bart_tokenizer


def summarize_clauses(flagged_clauses, model, tokenizer, device, batch_size=8, progress_callback=None):
    # flagged_clauses is a list of (clause, label) tuples. Batched so BART generates
    # several summaries per forward pass instead of one at a time. Batches are built
    # from clauses of similar length so little time goes into padding (same output,
    # ~30% faster); summaries are put back in the original order at the end
    model.eval()
    order = sorted(range(len(flagged_clauses)), key=lambda i: len(flagged_clauses[i][0]))
    summaries = []

    for i in range(0, len(order), batch_size):
        batch = [flagged_clauses[j] for j in order[i:i + batch_size]]

        # Match the input format BART was trained on: "{label}: {clause}"
        input_texts = [f"{label}: {clause}" for clause, label in batch]

        inputs = tokenizer(
            input_texts,
            max_length=BART_MAX_INPUT_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            # Generate summary
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=BART_MAX_TARGET_LENGTH,
                min_length=20,
                num_beams=4,        # beam search for better quality output
                length_penalty=2.0, # encourages longer summaries
                early_stopping=True
            )

        for ids in summary_ids:
            summary = tokenizer.decode(ids, skip_special_tokens=True)

            # Strip any label prefixes that leaked from placeholder training data
            for prefix in ['clearly_fair: ', 'clearly_unfair: ', 'potentially_unfair: ']:
                summary = summary.replace(prefix, '')

            summaries.append(summary.strip())

        if progress_callback:
            progress_callback(len(summaries), len(flagged_clauses))

    in_original_order = [None] * len(summaries)
    for j, summary in zip(order, summaries):
        in_original_order[j] = summary
    return in_original_order
