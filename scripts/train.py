import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.sec_to_sec import Seq2Seq
from src.data.loader import get_data_loaders
from src.model.encoder.encoder import Encoder
from src.model.decoder.decoder import Decoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---
CHECKPOINT_NAME = "transformer_model.pt"
TOKENIZER_NAME = "microsoft/codebert-base"
# Define the full path for the checkpoint
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "model", "outcome")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
HID_DIM = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 6
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(iterator, desc="Training")

    for batch in progress_bar:
        src = batch['input_ids'].to(DEVICE)
        trg = batch['labels'].to(DEVICE)

        trg_input = trg.clone()
        # trg_input[trg_input == -100] = 1
        trg_input[trg_input == -100] = tokenizer.pad_token_id

        optimizer.zero_grad()

        output, _ = model(src, trg_input[:, :-1])
        output_dim = output.shape[-1]

        trg_target = trg[:, 1:].contiguous().view(-1)
        output = output.contiguous().view(-1, output_dim)

        loss = criterion(output, trg_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src = batch['input_ids'].to(DEVICE)
            trg = batch['labels'].to(DEVICE)

            trg_input = trg.clone()
            trg_input[trg_input == -100] = 1

            output, _ = model(src, trg_input[:, :-1])

            output_dim = output.shape[-1]
            trg_target = trg[:, 1:].contiguous().view(-1)
            output = output.contiguous().view(-1, output_dim)

            loss = criterion(output, trg_target)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)

    INPUT_DIM = len(tokenizer)
    OUTPUT_DIM = len(tokenizer)
    SRC_PAD_IDX = tokenizer.pad_token_id
    TRG_PAD_IDX = tokenizer.pad_token_id

    # [FIX] Define the max length to match CodeBERT's limit (512)
    MAX_LEN = 512

    print(f"Vocab Size: {INPUT_DIM}, Pad IDX: {SRC_PAD_IDX}")

    # 2. Initialize Components
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, DROPOUT, DEVICE, max_length=MAX_LEN)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DROPOUT, DEVICE, max_length=MAX_LEN)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)

    # Apply init weights initially (will be overwritten if loading checkpoint)
    model.apply(init_weights)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- RESUME LOGIC STARTS HERE ---
    start_epoch = 0
    best_valid_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing checkpoint at {CHECKPOINT_PATH}")
        print("Loading model state and resuming training...")

        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

            # Load Model Weights
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load Optimizer State (Critical for AdamW momentum)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load Last Epoch and Best Loss
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint.get('loss', float('inf'))

            print(f"Successfully resumed from Epoch {start_epoch} (Last Valid Loss: {best_valid_loss:.4f})")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch instead.")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    # 3. Training Loop
    if start_epoch >= EPOCHS:
        print(f"Training already completed ({start_epoch}/{EPOCHS} epochs). Increase EPOCHS to train more.")
        return

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, 1.0)
        valid_loss = evaluate(model, val_loader, criterion)

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch + 1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')

        # --- SAVE LOGIC ---
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"\tNew best validation loss! (Saving checkpoint)")
        else:
            print(f"\tValidation loss did not improve. (Saving checkpoint to capture latest progress)")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_valid_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"\tCheckpoint updated at {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
