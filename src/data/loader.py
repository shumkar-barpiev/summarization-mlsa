import os
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq, AutoTokenizer


def get_data_loaders(
        data_path="./processed",
        batch_size=32,
        model_checkpoint="microsoft/codebert-base"
):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run the tokenization script first!")

    print(f"Loading dataset from {data_path}...")
    dataset = load_from_disk(data_path)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    print("Creating DataLoaders...")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data to prevent order bias
        collate_fn=data_collator,
        num_workers=2,  # Speed up loading (adjust based on CPU cores)
        pin_memory=True  # Speed up transfer to GPU
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    try:
        train_dl, val_dl, test_dl = get_data_loaders()
        print(f"Success! Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

        batch = next(iter(train_dl))
        print("Batch keys:", batch.keys())
        print("Input shape:", batch["input_ids"].shape)
    except Exception as e:
        print(f"Error: {e}")