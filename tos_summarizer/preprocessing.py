import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk_data():
    # One-time downloads for sentence splitting and POS tagging (has_verb).
    # The Docker image bakes these in at build time so this is a no-op there
    for resource, path in [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_df(df):
    df = df[df['sentence'].str.len() >= 20].copy()
    df['sentence'] = df['sentence'].str.strip()
    df = df[df['sentence'].str.len() > 0]
    return df


# to check if the snippet has a verb in it and filter out those that don't for better results
def has_verb(sentence):
    """Returns True if the sentence contains at least one verb —
    a decent proxy for 'this is a real clause, not a bare heading'."""
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    return any(tag.startswith('VB') for word, tag in tags)


def clean_tos_text(text):
    # Convert to DataFrame so we can run clean_df on it
    # We split on newlines to get individual lines first
    df = pd.DataFrame({'sentence': [s.strip() for s in text.split('\n') if len(s.strip()) > 0]})

    # Run our preprocessing cleaning function
    df = clean_df(df)

    # Return as a single clean string to pass into process_tos_document.
    # Lines are kept on separate rows (instead of joined with spaces) so a heading
    # like "Overview" doesn't get glued onto the start of the next sentence
    return '\n'.join(df['sentence'].tolist())


def load_tos_document(file_path):
    # .txt as is (same as pasting it into the app); .pdf, .docx and .html go through
    # extract.py, which turns them back into one paragraph per line
    if str(file_path).lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        from .extract import extract_text
        with open(file_path, 'rb') as f:
            text = extract_text(f.read(), str(file_path))

    return clean_tos_text(text)


def split_clauses(document, min_length=20):
    # Split document into individual clauses by sentence.
    # nltk's sent_tokenize instead of split('.') so "e.g.", "U.S.", "Inc." etc.
    # don't chop clauses in half. Each line is split separately so headings stay
    # on their own, then has_verb drops them since they aren't real clauses
    ensure_nltk_data()
    clauses = []
    for line in document.split('\n'):
        for sentence in sent_tokenize(line):
            sentence = sentence.strip()
            if len(sentence) >= min_length and has_verb(sentence):
                clauses.append(sentence)
    return clauses
