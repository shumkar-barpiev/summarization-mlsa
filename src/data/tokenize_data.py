import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from helper import preprocess_code

RAW_DATASET_NAME = "Nan-Do/code-search-net-python"
TOKENIZER_CHECKPOINT = "microsoft/codebert-base"
OUTPUT_PATH = "./processed"
MAX_LENGTH = 128

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize(examples, tokenizer):
    cleaned_codes = [preprocess_code(c, is_code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_code=False) for s in examples["summary"]]

    model_inputs = tokenizer(
        cleaned_codes,
        text_target=cleaned_summaries,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    return model_inputs


def main():
    print(f"Initializing tokenizer: {TOKENIZER_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)

    print(f"Loading raw dataset: {RAW_DATASET_NAME}...")
    try:
        # raw_dataset = load_dataset(RAW_DATASET_NAME)
        raw_dataset = load_dataset(RAW_DATASET_NAME, split='train[:10%]')
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print("Splitting dataset into Train (80%), Validation (10%), Test (10%)...")

    # train_test_valid = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_test_valid = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test_valid["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        'train': train_test_valid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

    print("Tokenizing and cleaning data (this may take a while)...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        num_proc=4,  # Use multiple CPU cores for speed
        # remove_columns=raw_dataset["train"].column_names
        remove_columns=raw_dataset.column_names
    )

    print(f"Saving processed dataset to {OUTPUT_PATH}...")
    tokenized_dataset.save_to_disk(OUTPUT_PATH)

    print("\nSuccess! Data Preparation Complete.")
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Val samples:   {len(tokenized_dataset['validation'])}")
    print(f"Test samples:  {len(tokenized_dataset['test'])}")


if __name__ == "__main__":
    main()
