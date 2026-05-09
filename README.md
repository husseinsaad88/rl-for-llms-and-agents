# RL for LLMs and Agents

A comprehensive guide to Reinforcement Learning for Large Language Models, demonstrating the complete pipeline from classical RLHF to modern RLVR (Reinforcement Learning with Verifiable Rewards), similar to techniques used in DeepSeek-R1.

## Quick Facts

- **Model**: Qwen2.5-1.5B-Instruct (quantized with 4-bit + LoRA for efficiency)
- **Task**: GSM8K (grade-school math word problems)
- **Training Data**: 500 problems | **Evaluation**: 200 held-out problems
- **Techniques**: QLoRA, PEFT, modern TRL library
- **License**: MIT

## What You'll Learn

This project demonstrates five distinct RL approaches for LLM training:

| Method | Approach | Key Feature |
|--------|----------|-------------|
| **SFT** | Supervised Fine-Tuning | Baseline: teach correct solutions |
| **RLHF (GRPO+RM)** | RL + Learned Reward Model | Classical approach: train neural network to score answers |
| **DPO** | Direct Preference Optimization | Skip the reward model; optimize directly from preferences |
| **RLVR** | RL with Verifiable Rewards | Use hard-coded correctness verification instead of neural RM |
| **Enhanced RLVR** | RLVR + Composite Rewards | Add bonuses for reasoning quality and chain-of-thought |

Each method progresses toward **better interpretability, efficiency, and performance** on the math reasoning task.

## Notebook Overview

Both notebooks follow an identical 8-section structure covering setup, baseline evaluation, and all five RL methods with final comparison.

### Version Comparison

| Aspect | v1.1 | v2 |
|--------|------|-----|
| Status | Original | **Recommended** ✓ |
| GRPO+RM Training | ~40 steps | 400 steps |
| Hyperparameters | Has temperature mismatch | Fixed |
| Results | Baseline results | Significantly improved |

**Use v2** — it includes corrected training hyperparameters and extended training runs, yielding substantially better results.

## Quick Start

### Prerequisites
- Python 3.8+
- GPU (Google Colab recommended for free access)
- Hugging Face account (to download Qwen model)

### Installation

```bash
pip install transformers trl peft bitsandbytes torch
pip install datasets evaluate scikit-learn matplotlib
```

### Running the Notebooks

1. **Open `rlhf_to_rlvr_complete_final_v2.ipynb`** (recommended version)
2. **In Google Colab**: 
   - Upload notebook or link from GitHub
   - Runtime → Change runtime type → GPU
   - Authenticate with Hugging Face when prompted
3. **Run sequentially through sections**:
   - **Section 0**: Setup & model loading
   - **Section 1**: Zero-shot baseline
   - **Sections 2-7**: Choose any RL method(s) to train
   - **Section 8**: Compare all results side-by-side

### What to Expect

- Training takes ~2-5 minutes per method (GPU-dependent)
- Outputs: Accuracy metrics, loss curves with rolling averages
- Visualization: Final comparison chart across all approaches
- Model saves: Adapters checkpoint to Google Drive (Colab) or local storage

## Project Structure

- **`rlhf_to_rlvr_complete_final_v2.ipynb`** — Main notebook (v2, recommended)
- **`rlhf_to_rlvr_complete_final_v1.1.ipynb`** — Reference version
- **`LICENSE`** — MIT License

Both notebooks are self-contained; no external scripts or data files required.

## Key Insights

- **RLVR is competitive**: Verifiable rewards match learned reward models without neural network training overhead
- **Emergent reasoning**: RL training encourages chain-of-thought behavior (e.g., "wait, let me recalculate...")
- **Method progression**: From classical RLHF → direct preference learning (DPO) → verifiable approaches (RLVR)
- **Memory efficiency**: QLoRA reduces model footprint by 4x while maintaining performance

## Contributing

This is an educational resource. Feel free to fork, modify, and extend for your own experiments!

## License

MIT License © 2026 husseinsaad88
