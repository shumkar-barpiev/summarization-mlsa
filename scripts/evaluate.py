import torch
import os
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
import evaluate as hf_evaluate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.sec_to_sec import Seq2Seq
from src.data.loader import get_data_loaders
from src.model.encoder.encoder import Encoder
from src.model.decoder.decoder import Decoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Config ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "model", "outcome", "transformer_model.pt")
TOKENIZER_NAME = "microsoft/codebert-base"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 50

# Hyperparameters (MUST MATCH TRAIN.PY)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
DROPOUT = 0.1


def translate_sentence(src_tensor, model, tokenizer, device, max_len=50):
    model.eval()

    # 1. Encode
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 2. Initialize Target with [SOS]
    trg_indices = [tokenizer.cls_token_id]  # Start with [CLS]/<s>

    # 3. Generate tokens one by one
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # Get prediction for the last token
        pred_token = output.argmax(2)[:, -1].item()

        trg_indices.append(pred_token)

        # Stop if we see [SEP]/</s>
        if pred_token == tokenizer.sep_token_id:
            break

    # Decode to string
    return tokenizer.decode(trg_indices, skip_special_tokens=True)


def main():
    # 1. Setup
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    _, _, test_loader = get_data_loaders(batch_size=1)  # Batch size 1 for qualitative check

    INPUT_DIM = len(tokenizer)
    OUTPUT_DIM = len(tokenizer)
    SRC_PAD_IDX = tokenizer.pad_token_id
    TRG_PAD_IDX = tokenizer.pad_token_id

    # 2. Rebuild Model Structure
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, DROPOUT, DEVICE)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DROPOUT, DEVICE)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)

    # 3. Load Weights
    print(f"Loading weights from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Train the model first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # 4. Evaluation Loop
    bleu_metric = hf_evaluate.load("bleu")
    predictions = []
    references = []

    print("Generating summaries...")
    limit = 100  # Limit to 100 samples for speed

    for i, batch in enumerate(tqdm(test_loader)):
        if i >= limit: break

        src = batch['input_ids'].to(DEVICE)
        labels = batch['labels']

        # Replace -100 with pad token for decoding reference
        labels[labels == -100] = tokenizer.pad_token_id

        # Generate
        pred_str = translate_sentence(src, model, tokenizer, DEVICE, MAX_LEN)
        ref_str = tokenizer.decode(labels[0], skip_special_tokens=True)

        predictions.append(pred_str)
        references.append([ref_str])  # BLEU expects list of references

        # Print a few examples
        if i < 3:
            print(f"\n--- Example {i + 1} ---")
            print(f"Ref : {ref_str}")
            print(f"Pred: {pred_str}")

    # 5. Calculate Metrics
    results = bleu_metric.compute(predictions=predictions, references=references)
    print(f"\nFinal BLEU Score: {results['bleu']:.4f}")


if __name__ == "__main__":
    main()