"""Stage 1 — load TOS_DatasetV3, clean it, and save the splits to CSV.
(From notebooks/ToS_preprocessing.ipynb)

    python -m training.preprocess
"""
import pandas as pd
from datasets import load_dataset

from tos_summarizer.config import DATA_DIR, label2id
from tos_summarizer.preprocessing import clean_df


def main():
    # Load the dataset
    dataset = load_dataset("CodeHima/TOS_DatasetV3")

    # Properly split the dataset into train, validation, and test sets
    df_train = pd.DataFrame(dataset['train'])
    df_val = pd.DataFrame(dataset['validation'])
    df_test = pd.DataFrame(dataset['test'])

    # Clean up the dataframes for each split
    df_train = clean_df(df_train)
    df_val = clean_df(df_val)
    df_test = clean_df(df_test)

    # Apply label mapping to all three splits
    df_train['label'] = df_train['unfairness_level'].map(label2id)
    df_val['label'] = df_val['unfairness_level'].map(label2id)
    df_test['label'] = df_test['unfairness_level'].map(label2id)

    # Sanity check the label distribution in training set
    print(df_train['label'].value_counts())
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Save the pandas dfs to csv so they can be used in the other stages
    df_train.to_csv(DATA_DIR / 'df_train.csv', index=False)
    df_val.to_csv(DATA_DIR / 'df_val.csv', index=False)
    df_test.to_csv(DATA_DIR / 'df_test.csv', index=False)


if __name__ == "__main__":
    main()
