import torch
import os
import sys
import evaluate  # Hugging Face evaluate library
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.model.sec_to_sec import Seq2Seq
from src.data.loader import get_data_loaders
from src.model.encoder.encoder import Encoder
from src.model.decoder.decoder import Decoder

TOKENIZER_NAME = "microsoft/codebert-base"
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "model", "outcome", "transformer_model.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (Ensure these match train.py exactly)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
DROPOUT = 0.1
MAX_LEN = 512  # Match the value used in train.py


def translate_sentence(sentence_tensor, model, device, max_len=100, sos_idx=0, eos_idx=2):
    model.eval()

    with torch.no_grad():
        src_mask = model.make_src_mask(sentence_tensor)
        enc_src = model.encoder(sentence_tensor, src_mask)

    # 2. Initialize Target with <sos>
    trg_indexes = [sos_idx]

    # 3. Generate tokens one by one
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # Get the prediction for the last token
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == eos_idx:
            break

    return trg_indexes[1:]  # Skip <sos>


def main():
    print(f"Running evaluation on device: {DEVICE}")

    # 1. Load Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # We only need the test loader here
    _, _, test_loader = get_data_loaders(batch_size=1)

    INPUT_DIM = len(tokenizer)
    OUTPUT_DIM = len(tokenizer)

    # Special Tokens
    SOS_IDX = tokenizer.cls_token_id  # usually 0 for RoBERTa/CodeBERT
    EOS_IDX = tokenizer.sep_token_id  # usually 2
    PAD_IDX = tokenizer.pad_token_id  # usually 1

    print(f"Vocab: {INPUT_DIM}, SOS: {SOS_IDX}, EOS: {EOS_IDX}, PAD: {PAD_IDX}")

    # 2. Initialize Model
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, DROPOUT, DEVICE, max_length=MAX_LEN)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DROPOUT, DEVICE, max_length=MAX_LEN)
    model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Handle 'model_state_dict' wrapper if present
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"ERROR: No model found at {MODEL_PATH}")
        return

    # 4. Evaluation Loop (BLEU)
    bleu_metric = evaluate.load("bleu")

    predictions = []
    references = []

    print("Starting generation...")
    # Limit to first 100 examples for speed if needed, otherwise remove [:100]
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        src = batch['input_ids'].to(DEVICE)
        trg = batch['labels'].to(DEVICE)  # True summary

        # Generate Summary
        translation_ids = translate_sentence(src, model, DEVICE, max_len=50, sos_idx=SOS_IDX, eos_idx=EOS_IDX)

        # Decode IDs to Text
        pred_text = tokenizer.decode(translation_ids, skip_special_tokens=True)

        # For references, we must handle the batch dimension and remove -100 if present
        # trg is shape [1, seq_len] since batch_size=1
        trg_list = trg.squeeze().tolist()
        # Filter out -100 and padding
        clean_trg = [t for t in trg_list if t != -100 and t != PAD_IDX]
        ref_text = tokenizer.decode(clean_trg, skip_special_tokens=True)

        predictions.append(pred_text)
        references.append([ref_text])  # BLEU expects a list of references for each prediction

        # Optional: Print first few examples to verify
        if i < 3:
            print(f"\nExample {i}:")
            print(f"Ref:  {ref_text}")
            print(f"Pred: {pred_text}")

    # 5. Compute Score
    print("\nComputing BLEU Score...")
    results = bleu_metric.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {results['bleu'] * 100:.2f}")


if __name__ == "__main__":
    main()