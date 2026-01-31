# AI Code Summarizer: Sequence-to-Sequence Transformer

This project implements a **Transformer-based Encoder-Decoder architecture** from scratch to perform automatic source code summarization. The model takes Python function definitions as input and generates human-readable English docstrings (summaries).

It handles the full machine learning pipeline: from data tokenization (using CodeBERT) to training a custom Transformer model and evaluating it using BLEU scores.

## 📂 Project Structure

The repository is organized as follows:

```text
├── README.md                 # Project documentation
├── summarization_mlsa.ipynb  # For quick model inference and demo
├── scripts
│   ├── run_evaluation.py     # Script to calculate BLEU scores on test data
│   ├── summarize.py          # Inference script to test single code snippets
│   └── train.py              # Main training loop
└── src
    ├── data
    │   ├── helper.py         # Helper methods contained for cleaning the data
    │   ├── loader.py         # Data loading utilities
    │   ├── tokenize_data.py  # Preprocessing and tokenization script
    │   └── processed/        # (Created after tokenization) Stores processed data
    └── model
        ├── encoder/          # Transformer Encoder implementation
        ├── decoder/          # Transformer Decoder implementation
        └── sec_to_sec.py     # Seq2Seq wrapper
```

### 1. Prerequisites
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)

### 2. Installation
Clone the repository and install the required dependencies:

```
git clone https://github.com/shumkar-barpiev/summarization-mlsa.git
cd summarization-mlsa
pip install torch torchvision torchaudio transformers datasets
```

## 🛠️ Usage Workflow
To train the model from scratch, you must follow these steps in order.

### Step 1: Data Preparation
Before training, the dataset (**CodeSearchNet**) must be downloaded and tokenized. We use the **CodeBERT** tokenizer (microsoft/codebert-base) for efficient subword tokenization.

Run the tokenization script:
```
python src/data/tokenize_data.py
```
- Output: This will create a 'src/data/processed/' directory containing the tokenized arrow files for train, validation, and test sets.

### Step 2: Training the Model
Once the data is ready, you can start the training process.

```
python scripts/train.py
```
## ⚠️ HARDWARE WARNING: USE A GPU

Training a Transformer model is computationally intensive.

- CPU: Training will likely take days and is not recommended.

- GPU (Recommended): On a Tesla P100 or T4 (available on Kaggle/Colab), training takes approximately 3-4 hours for 10 epochs.
`However, this depends heavily on GPU capacity—higher-end cards will complete the task much faster.`

If you are using `Google Colab` or `Kaggle`, make sure to enable the GPU Accelerator in your runtime settings.

### Step 3: Evaluation
After training, evaluate the model's performance on the unseen test set. This script calculates the BLEU score and Loss.

```
python scripts/run_evaluation.py
```

### Step 4: Inference (Demo)
To test the model with your own Python code, use the inference script.

```
python scripts/summarize.py "def add(a, b): return a + b"
```

### Example Output:
```
Code Input: def add(a, b): return a + b
Generated Summary: Adds two numbers and returns the result.
```

## Model Architecture
The project utilizes a standard Seq2Seq Transformer architecture:

- Encoder: Processes the source code tokens using Self-Attention to capture syntax and semantic dependencies.
- Decoder: Generates the English summary autoregressively, attending to the Encoder's output.

### Hyperparameters:

- Embedding Size: 64 (for lightweight testing)
- Attention Heads: 8
- Layers: 1 (Encoder) / 1 (Decoder)
- Dropout: 0.1

## 📊 Results

- Epoch: 01 | Time: 9m 2s
		Train Loss: 1.005 | Validation Loss: 0.895
- Epoch: 02 | Time: 9m 3s
		Train Loss: 0.998 | Validation Loss: 0.882
- Epoch: 03 | Time: 9m 5s
		Train Loss: 0.815 | Validation Loss: 0.640
- Epoch: 04 | Time: 9m 5s
		Train Loss: 0.666 | Validation Loss: 0.512
- Epoch: 05 | Time: 9m 6s
		Train Loss: 0.656 | Validation Loss: 0.498
- Epoch: 06 | Time: 9m 8s
		Train Loss: 0.654 | Validation Loss: 0.495
- Epoch: 07 | Time: 9m 4s
		Train Loss: 0.461 | Validation Loss: 0.367
- Epoch: 08 | Time: 9m 6s
		Train Loss: 0.368 | Validation Loss: 0.350
- Epoch: 09 | Time: 9m 9s
		Train Loss: 0.346 | Validation Loss: 0.342
- Epoch: 10 | Time: 9m 11s
		Train Loss: 0.338 | Validation Loss: 0.338

> **Training Loss**: Converged to ~0.338

> **Validation Loss**: Converged to ~0.338

> **BLEU Score**: ~7.05 (demonstrating semantic understanding of code intent)


## 🌐 Quick Start: Run Online (No Installation)

You can test the trained model immediately in your browser using Google Colab, without downloading the repository or setting up a local environment.

1.  Open **[Google Colab](https://colab.research.google.com/)**.
2.  Go to **File > Open Notebook**.
3.  Select the **GitHub** tab.
4.  Paste the link to the notebook:
    ```text
    https://github.com/shumkar-barpiev/summarization-mlsa/blob/main/summarization_mlsa.ipynb
    ```
5.  Run the cells. The notebook will automatically set up the environment and run the demo.

---

## Data & Model Availability

To keep this repository lightweight, it contains **source code only**. The large datasets and trained model weights are stored externally.

* **Full Project Archive (Model + Dataset):** [Download from Google Drive](https://drive.google.com/file/d/1WI_KNIDUM6LAocZ36_x1aNXMbsAEfzff/view?usp=drive_link)

> **Note:** You do **not** need to download this manually if you are using the Colab notebook mentioned above; the notebook automates the download process for you.

