# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanoPoet** is an educational Chinese classical poetry generation project that demonstrates the complete training pipeline of a GPT-style language model on a personal computer. The project trains a model to generate classical Chinese poetry based on specified conditions (author style, poetry format, title).

Key constraints:
- Designed for low-resource environments (MacBook CPU/MPS)
- Small dataset and model size (not suitable for chat-style interaction)
- Character-level tokenizer with ~11,868 vocab size
- Supports Tang poetry (唐诗), Song poetry/ci (宋词), multiple formats

## Project Structure

```
nanopoet/
├── raw/                          # Raw data: base_poetry_data.jsonl (~240K poems)
├── nanopoet/                     # Main package
│   ├── main.py                   # Entry point with training orchestration
│   ├── model.py                  # GPT model architecture (Transformer blocks)
│   ├── common.py                 # Special tokens, CharTokenizer, encoding utilities
│   ├── dataset.py                # Data loading and splitting functions
│   ├── train_pre.py              # Pre-training stage
│   ├── train_mid.py              # Mid-training stage with metadata
│   ├── train_sft.py              # Supervised fine-tuning stage
│   ├── train_reward.py           # Reward classifier training (binary author-style matching)
│   ├── train_rl.py               # RL training with REINFORCE algorithm
│   ├── reward.py                 # Reward computation functions (rhyme, format, author-style)
│   └── app.py                    # Flask web application for generation
├── notebook/                     # Jupyter notebooks for exploration
│   ├── 00_prepare.ipynb         # Data analysis and tokenizer design
│   ├── 01_model.ipynb           # Model architecture exploration
│   ├── 02_pre_train.ipynb       # Pre-training experiments
│   ├── 03_mid_train.ipynb       # Mid-training experiments
│   ├── 04_sft.ipynb             # SFT experiments
│   ├── 05_reward.ipynb          # Reward classifier training
│   └── 06_rl.ipynb              # RL training experiments
└── archive/                      # Legacy code from previous iterations
```

## Training Pipeline

The training follows a 5-stage pipeline inspired by [nanochat](https://github.com/karpathy/nanochat):

### 1. Pre-Training (PRE)
- **Purpose**: Learn basic language patterns and poetry structure
- **Data**: Concatenated poetry content only (no metadata)
- **Data size**: ~11M tokens from all 240K poems
- **Training**: Standard next-token prediction on raw content
- **Output**: `output/pre_train_model.pt`

### 2. Mid-Training (MID)
- **Purpose**: Learn structured poetry format with metadata
- **Data**: Full poems with special tokens encoding author, style, title, content
- **Data size**: ~16M tokens with metadata markers
- **Special tokens**: `B` (begin), `A...a` (author), `S...s` (style), `T...t` (title), `C...c` (content), `P` (padding)
- **Key difference**: Uses padding and masking for variable-length sequences
- **Learning rate**: Starts at 50% of base LR, warmdown in final 20% of training
- **Output**: `output/mid_train_model.pt`

### 3. Supervised Fine-Tuning (SFT)
- **Purpose**: Condition on metadata to generate poetry matching specific author/style
- **Data**: Filtered subset of ~24K poems from 15 famous authors and specific formats
- **Supported authors**: 苏轼, 辛弃疾, 李清照, 柳永, 欧阳修, 李白, 杜甫, 白居易, 王维, 李商隐, 陆游, 杨万里, 黄庭坚, 王安石, 朱熹
- **Supported styles**: 七言律诗, 七言绝句, 五言律诗, 五言绝句, 浣溪沙, 水调歌头, 西江月, 鹧鸪天, 沁园春, 蝶恋花
- **Key technique**: Random metadata erasure during training (50% probability per field)
- **Loss calculation**: Only compute loss on content tokens (between `C` and `c`), ignore metadata using `ignore_index=-1`
- **Learning rate**: Starts at 2% of base LR, linear decay to 0
- **Output**: `output/sft_model.pt`

### 4. Reward Classifier Training
- **Purpose**: Train a binary classifier to judge author-style matching
- **Architecture**: Adds bidirectional Transformer layers on top of frozen SFT model
- **Data**: Uses ~24K filtered poems from 15 target authors
- **Sample generation strategy**:
  - All filtered poems as positive samples (original author)
  - Equal number of negative samples (random author swap)
- **Training**: Only trains bidirectional layers + classification head (~1.6M params), GPT weights frozen
- **Output**: `output/reward_classifier.pt`

### 5. Reinforcement Learning (RL)
- **Purpose**: Further refine quality through reward-based training with REINFORCE algorithm
- **Algorithm**: Policy gradient (REINFORCE) with baseline to reduce variance
- **Three reward objectives**:
  - Rhyme checking (30%): Based on pypinyin, checks if 70% of sentence endings share same rhyme
  - Format checking (30%): Strict matching against extracted format patterns
  - Author-style matching (40%): Neural classifier from stage 4
- **Key techniques**:
  - Sample multiple outputs per prompt (8 samples default)
  - Advantage = reward - mean(rewards) to reduce variance
  - Only compute gradient on generated content, not prompt
  - Learning rate linear decay over training
  - Moving average (50-step window) to track training trends
- **Training config**: `num_steps=1000`, `lr=1e-5`, `temperature=1.0`, `top_k=50`
- **Output**: `output/rl_model.pt`

## Model Architecture

**GPTLanguageModel** (nanopoet/model.py:107):
- Decoder-only Transformer architecture
- Default configuration:
  - `vocab_size`: 11,868 (character-level)
  - `emb_size`: 256
  - `block_size`: 256 (max context length)
  - `layer_num`: 8 (Transformer blocks)
  - `head_num`: 8 (attention heads)
  - `dropout`: 0.1 (training), 0.0 (inference)
- Key features:
  - Parallel multi-head attention (optimized)
  - Causal masking for autoregressive generation
  - Generation with temperature, top-k sampling, and stop tokens

**BinaryClassifier** (nanopoet/train_reward.py):
- Architecture: Frozen GPT + Bidirectional Transformer + Classification Head
- Purpose: Judge if author and poem style match
- Components:
  - Base GPT (frozen): Uses embeddings and representations from SFT model
  - Bidirectional layers: 2 Transformer blocks without causal masking
  - Global pooling: Average across all sequence positions (not just last token)
  - Classification head: Linear layer → sigmoid for binary classification
- Training: Only bidirectional layers and classification head are trainable (~1.6M params)
- Why bidirectional: Author-style matching requires understanding full context, not just left-to-right

## Common Development Commands

### Setup and Dependencies
```bash
# Install dependencies (using uv)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training Stages
```bash
# Run from nanopoet/ directory
cd nanopoet

# Run specific training stage (edit main.py to uncomment desired stage)
# Pre-training
python main.py  # Uncomment line 316

# Mid-training
python main.py  # Uncomment line 320

# SFT
python main.py  # Uncomment line 324

# Reward classifier
python main.py  # Uncomment line 328

# RL training
python main.py  # Uncomment line 332

# Start web app
python main.py  # Uncomment line 335
```

### Web Application
```bash
# Start Flask web app (runs on port 54321)
python -m nanopoet.main  # After uncommenting start_app() in main.py

# Or run app.py directly (not recommended, use main.py)
```

### Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Navigate to notebook/ directory and open desired notebook
```

## Key Implementation Details

### Special Token Encoding
Poems are encoded with structural markers (common.py:52):
```
B + A<author>a + S<style>s + T<title>t + C<content>c
```

Example: `BA陆游aS五言律诗sT送仲高兄宫學秩滿赴行在tC兄去游東閤，才堪直北扉...c`

### Data Splitting (dataset.py:20)
- 9:1 train/val split using stratified sampling
- Sorts by style first, then every 10th poem goes to validation
- Ensures balanced style distribution across splits

### Checkpoint Management
All training stages support:
- Automatic checkpoint saving per epoch: `checkpoints/{stage}/epoch_XXX.pt`
- Resume training from latest checkpoint automatically
- Final model saved to: `output/{stage}_model.pt`

### Training Hyperparameters
Typical values from main.py:
- Pre-training: `batch_size=64`, `lr=3e-4`, `epochs=10`
- Mid-training: `batch_size=32`, `lr=3e-4`, `init_lr_frac=0.5`, `epochs=10`
- SFT: `batch_size=16`, `lr=3e-4`, `init_lr_frac=0.2`, `epochs=10`
- Reward classifier: `batch_size=16`, `lr=1e-4`, `epochs=10`
- RL training: `num_samples_per_prompt=8`, `num_steps=1000`, `lr=1e-5`
- All stages use `grad_clip=1.0`

### Device Selection (main.py:151)
Automatic device priority: CUDA > MPS > CPU
```python
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
```

## Architecture Notes

### Why Character-Level Tokenizer?
- Dataset contains only Chinese characters, Chinese punctuation (。，), and special marker letters
- Total unique characters: ~11,857
- Simple implementation, no need for BPE training
- Special tokens (A, a, S, s, T, t, C, c, B, P, U) are ASCII letters not present in Chinese text

### Loss Masking Strategy
- **Mid-training**: Mask padding tokens only (mask=1 for real tokens, mask=0 for padding)
- **SFT**: Use `ignore_index=-1` to mask both padding and metadata, compute loss only on content between `C` and `c` tokens
- **RL training**: Only compute policy gradient on generated content tokens (not prompt), use mask to exclude prompt from gradient calculation

### Reward Computation (reward.py)
Three reward objectives combined with weighted sum:
1. **Rhyme checking** (`check_rhyme`):
   - Uses `pypinyin` to extract rhyme (finals) from sentence-ending characters
   - Requires 70% of sentences to share the most common rhyme
   - Returns 0-1 score based on rhyme consistency

2. **Format checking** (`check_format`):
   - Generates format code from content (e.g., "7-7-7-7" for 七言绝句)
   - Strict binary match: 1 if format matches extracted patterns, 0 otherwise
   - Format patterns extracted from training data via `extract_format`

3. **Author-style matching** (neural classifier):
   - Uses BinaryClassifier with frozen GPT + bidirectional Transformer
   - Global average pooling across sequence for classification
   - Returns probability (0-1) of author-poem match

Combined reward: `0.3 * rhyme + 0.3 * format + 0.4 * author_style`

### REINFORCE Algorithm Implementation (train_rl.py)
Key steps per training iteration:
1. Generate prompt with random author/style (50% probability each)
2. Sample multiple completions (8 by default) using top-k sampling
3. Compute rewards for each sample using `compute_reward`
4. Calculate advantage: `advantage = reward - mean(rewards)` (batch-level baseline)
5. Compute policy gradient loss: `Loss = -Σ log_prob * advantage` (only on generated tokens)
6. Backprop and update with gradient clipping
7. Track moving average (50-step window) to monitor training trends

### Learning Rate Schedules
- **Pre-training**: Constant learning rate
- **Mid-training**: Constant for 80%, linear warmdown in final 20%
- **SFT**: Linear decay from start (encourages stability when fine-tuning on smaller dataset)
- **RL training**: Linear decay from start to end (`lr_mult = 1.0 - step / num_steps`)

## Output Directory Structure

```
output/  # or $NANOPOET_BASE_DIR if set
├── checkpoints/
│   ├── pre/
│   │   └── epoch_XXX.pt
│   ├── mid/
│   │   └── epoch_XXX.pt
│   ├── sft/
│   │   └── epoch_XXX.pt
│   ├── reward/
│   │   └── epoch_XXX.pt
│   └── rl/
│       └── step_XXX.pt
├── pre_train_model.pt
├── mid_train_model.pt
├── sft_model.pt
├── reward_classifier.pt
└── rl_model.pt
```

## Web Application

The Flask app (app.py) provides:
- Model selection dropdown (automatically loads available trained models)
- Input fields: Author (optional), Style (optional), Title (optional)
- Generation parameters: Temperature (0.1-2.0), Top-K (1-100), Max Length
- Supports 15 authors and 10 poetry styles from SFT training
- Generates poetry matching specified constraints

## References

This project is inspired by:
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) - Dataset source
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Training approach
- [nanochat](https://github.com/karpathy/nanochat) - Multi-stage training pipeline
- [Andrej Karpathy - GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Educational foundation