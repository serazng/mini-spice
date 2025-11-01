# Mini-SPICE

Minimal, single-GPU/CPU implementation of
[SPICE: Self-Play in Corpus Environments Improves Reasoning](https://arxiv.org/abs/2510.24684) [[1]](#references)

## Overview

SPICE is a self-play reinforcement-learning framework in which a single model plays two complementary roles:

Challenger: Mines documents from a corpus to generate diverse, verifiable reasoning tasks.
Reasoner: Solves these tasks without access to the documents.

The key idea is a Gaussian reward centered around a 50 % pass-rate. This drives the Challenger to produce questions that are neither trivial nor impossible—an emergent curriculum that helps the Reasoner continually improve.
Both agents are optimized via GRPO-style policy updates with centered advantages.

## Quick Start

```bash
# Install
pip install -e .
pip install torch transformers accelerate sympy datasets numpy pandas bitsandbytes safetensors peft

# Prepare corpus
prepare-corpus --in ./my_docs --out mini_spice/data/corpus

# Train
run-train

# Evaluate
run-eval --checkpoint checkpoints/step_100/ --eval-file mini_spice/eval/datasets/gsm8k-mini.json
```

## Configuration

Override defaults via CLI:

```bash
run-train \
    --T 200 \
    --B 16 \
    --G 4 \
    --temp-C 1.0 \
    --temp-R 1.0 \
    --4bit \
    --lora
```

Key parameters:
- `T`: Training iterations (default: 100)
- `B`: Batch size (default: 8)
- `G`: Reasoner attempts per Challenger question (default: 3)
- `temp-C`, `temp-R`: Sampling temperatures
- `--4bit`, `--lora`: Memory-efficient training

### Environment Variables

Set directory paths via environment variables (`.env`):

- `MINI_SPICE_RUNS_DIR`: Directory for training logs (default: `runs`)
- `MINI_SPICE_CHECKPOINTS_DIR`: Directory for checkpoints (default: `checkpoints`)
- `MINI_SPICE_CORPUS_DIR`: Corpus directory (default: `mini_spice/data/corpus`)

## Project Structure

```
mini_spice/
├── roles.py          # Challenger/Reasoner prompts
├── verifier.py       # Type-aware answer verification
├── rewards.py        # Gaussian reward for Challenger
├── policy.py         # GRPO updates
├── trainer.py        # Self-play training loop
├── storage.py        # Logging utilities
└── config.py         # Configuration
```

## Expected Results

- ≥ 60 % Challenger validity by step ≈ 50
- Pass-rate p stabilizes near 0.5 (curriculum signal)
- Post-training evaluation shows +2–5 pp accuracy gain on held-out reasoning sets

### Testing

```bash
pytest -q
```

## References

[1] Liu, B., Jin, C., Kim, S., Yuan, W., Zhao, W., Kulikov, I., Li, X., Sukhbaatar, S., Lanchantin, J., & Weston, J. (2025). SPICE: Self-Play In Corpus Environments Improves Reasoning. arXiv preprint arXiv:2510.24684. https://arxiv.org/abs/2510.24684