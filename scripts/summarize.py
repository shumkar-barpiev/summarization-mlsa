import torch
import os
import sys
import argparse
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model.sec_to_sec import Seq2Seq
    from src.model.encoder.encoder import Encoder
    from src.model.decoder.decoder import Decoder
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules from 'src'.")
    print(f"Detailed Error: {e}")
    sys.exit(1)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

MODEL_PATH = os.path.join(project_root, "src", "model", "outcome", "transformer_model.pt")
TOKENIZER_NAME = "microsoft/codebert-base"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HYPERPARAMETERS  ---
HID_DIM = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
DROPOUT = 0.1
MAX_LEN = 512

def load_model():
    """Loads the tokenizer and the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    enc = Encoder(len(tokenizer), HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, DROPOUT, DEVICE, MAX_LEN)
    dec = Decoder(len(tokenizer), HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DROPOUT, DEVICE, MAX_LEN)

    pad_idx = tokenizer.pad_token_id
    model = Seq2Seq(enc, dec, pad_idx, pad_idx, DEVICE).to(DEVICE)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"\nError loading weights! Ensure HID_DIM matches training (Currently: {HID_DIM})")
        print(f"Details: {e}")
        sys.exit(1)

    model.eval()
    return model, tokenizer

def predict(model, tokenizer, source_code, max_output_len=50):
    """Generates a summary for the given source code."""
    tokens = tokenizer.tokenize(source_code)[:MAX_LEN - 2]
    ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]

    src_tensor = torch.LongTensor(ids).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [tokenizer.cls_token_id]

    for i in range(max_output_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        if pred_token == tokenizer.sep_token_id:
            break

        trg_indexes.append(pred_token)

    summary = tokenizer.decode(trg_indexes[1:], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description='Generate a summary for Python source code.')
    parser.add_argument('input_code', type=str, help='The Python code string to summarize')
    args = parser.parse_args()

    model, tokenizer = load_model()
    summary = predict(model, tokenizer, args.input_code)

    print("\n------------------------------------------------")
    print("Code Input:")
    print(f"{args.input_code}")
    print("------------------------------------------------")
    print(f"Generated Summary: \033[92m{summary}\033[0m")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    main()