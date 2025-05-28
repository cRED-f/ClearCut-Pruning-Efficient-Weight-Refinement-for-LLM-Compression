# ClearCut Pruning: Efficient Weight Refinement for LLM Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of post-training pruning methods for large language models (LLMs), including our novel **ClearCut Pruning** algorithm and integration with low-rank decomposition.

## ✨ Key Features

- **Multiple pruning algorithms** - Magnitude, Wanda, SparseGPT, RIA, and our novel ClearCut method
- **N:M structured sparsity** - Hardware-friendly patterns (2:4, 1:4, 3:4, 4:8)
- **Low-rank decomposition** - Attention layer compression with SVD

## 📊 Results (50% Sparsity on WikiText2)

| Model        | SparseGPT | Wanda | RIA   | **ClearCut(Ours)** |
| ------------ | --------- | ----- | ----- | -------------- |
| LLaMA-2-7B   | 7.24      | 7.26  | 6.81  | **6.78**       |
| LLaMA-3.1-8B | -         | -     | 9.43  | **9.39**       |
| OPT-6.7B     | 11.55     | 11.94 | 11.84 | **11.44**      |

_Lower perplexity = better performance_

## 🛠️ Implemented Methods

### 1. ClearCut Pruning

Our novel method combining local weight importance with global activation statistics:

```python
def compute_clearcut_score(W, row_sum, col_sum, activation_norm, alpha=0.1, eps=1e-8):
    """
    Compute ClearCut importance scores for weight matrix W
    """
    abs_W = torch.abs(W)

    # Local importance components
    col_term = abs_W / (col_sum + eps)
    row_term = abs_W / (row_sum + eps)

    # Global interaction term
    interaction = alpha * (abs_W**2) / (row_sum * col_sum + eps)

    # Activation-aware adjustment
    activation_weight = activation_norm ** args.a

    return (col_term + row_term + interaction) * activation_weight
```

### 2. Low-Rank Decomposition

SVD-based compression for attention layers:

```python
def apply_low_rank_decomposition(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], rank_ratio=0.5):
    # Perform SVD on weight matrices
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Truncate to target rank
    target_rank = int(min_dim * rank_ratio)
    U_r, S_r, Vh_r = U[:, :target_rank], S[:target_rank], Vh[:target_rank, :]

    # Create factorized matrices
    A = U_r @ torch.diag(torch.sqrt(S_r))
    B = torch.diag(torch.sqrt(S_r)) @ Vh_r

    return LowRankLinear(A, B)
```

## 📂 Repository Structure

```
├── main.py              # Main entry point for pruning experiments
├── prune.py             # Core pruning algorithms (ClearCut, RIA, magnitude, wanda, sparsegpt)
├── data.py              # Dataset loading utilities (C4, WikiText2, PTB)
├── eval.py              # Evaluation scripts for perplexity and zero-shot tasks
├── layerwrapper.py      # Layer wrapping utilities for activation collection
├── sparsegpt.py         # SparseGPT implementation
├── quant.py             # Quantization utilities
└── README.md           # This file
```

## 🎯 Evaluation

### Perplexity Evaluation

```bash
# Evaluate on WikiText2
python main.py --model meta-llama/Llama-2-7b-hf --eval_dataset wikitext2

# Evaluate on multiple datasets
python main.py --model facebook/opt-6.7b --eval_dataset c4
```

### Zero-shot Task Evaluation

```bash
# Run zero-shot evaluation on common sense reasoning tasks
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method ClearCut \
    --sparsity_type 2:4 \
    --eval_zero_shot
```

Supported zero-shot tasks: BoolQ, RTE, HellaSwag, ARC-Challenge, MNLI

## 💾 Installation

```bash
git clone https://github.com/cRED-f/ClearCut-Pruning-Efficient-Weight-Refinement-for-LLM-Compression.git
```

**Dependencies:**

- PyTorch >= 2.0 (with semi-structured sparse support)
- Transformers >= 4.21.0
- Datasets
- SciPy (for Linear Sum Assignment)
- NumPy

## 🚀 Quick Start

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from main import main
import argparse

# Load model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure pruning arguments
args = argparse.Namespace(
    model=model_name,
    prune_method="ClearCut",          # Choose: magnitude, wanda, ria, ClearCut, sparsegpt
    sparsity_type="2:4",         # unstructured, 1:4, 2:4, 3:4, 4:8
    sparsity_ratio=0.5,          # For unstructured pruning
    calib_dataset="c4",          # c4, wikitext2, ptb
    nsamples=128,
    seqlen=2048,
    reallocation=True,           # Enable channel reallocation
    lsa=True,                    # Enable Linear Sum Assignment
    apply_low_rank=True,         # Apply low-rank decomposition
    rank_ratio=0.5,              # Rank ratio for SVD
    target_modules="q,k,v,o"     # Attention modules for low-rank
)

# Run pruning
main(args)
```

### Command Line Usage

```bash
# ClearCut Pruning with 2:4 sparsity
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method ClearCut \
    --sparsity_type 2:4 \
    --calib_dataset c4 \
    --reallocation \
    --lsa \
    --save

# RIA Pruning with unstructured 50% sparsity
python main.py \
    --model facebook/opt-6.7b \
    --prune_method ria \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --reconstruction \
    --save_model ./pruned_model

# Combined Pruning + Low-rank Decomposition
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method ClearCut \
    --sparsity_type 2:4 \
    --apply_low_rank \
    --rank_ratio 0.3 \
    --target_modules q,k,v,o \
    --eval_zero_shot
```

### Key Arguments

- `--prune_method`: Choose from `magnitude`, `wanda`, `ria`, `ClearCut`, `sparsegpt`
- `--sparsity_type`: `unstructured`, `1:4`, `2:4`, `3:4`, `4:8`
- `--sparsity_ratio`: Sparsity level for unstructured pruning (0.0-1.0)
- `--reallocation`: Enable heuristic channel reallocation
- `--lsa`: Enable Linear Sum Assignment optimization
- `--reconstruction`: Use SparseGPT-based weight reconstruction
- `--semi_sparse_acc`: Use PyTorch 2:4 semi-structured acceleration
- `--apply_low_rank`: Apply low-rank decomposition after pruning
- `--rank_ratio`: Fraction of singular values to keep (0.0-1.0)

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
