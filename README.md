## 🎯 Project Overview

This project implements and compares multiple transformer architectures for:
- **Text → Gloss Translation**: Convert English text to ASL gloss notation
- **Gloss → Text Translation**: Convert ASL gloss notation to English text

### Key Features

- ✅ **Baseline Transformer**: Standard seq2seq transformer (Vaswani et al., 2017)
- ✅ **Modern Transformer**: State-of-the-art architecture with:
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA)
  - RMSNorm (Root Mean Square Normalization)
  - SwiGLU Activation Functions
- ✅ **Bidirectional Training**: Both text→gloss and gloss→text directions
- ✅ **Comprehensive Evaluation**: BLEU, METEOR, chrF++, ROUGE metrics
- ✅ **Modular Architecture**: Easy to extend and experiment

## 📊 Results

Current performance on ASLG-PC12 dataset:

| Model | Direction | BLEU-4 | METEOR | chrF++ | ROUGE-L |
|-------|-----------|--------|--------|--------|---------|
| Baseline | Text→Gloss | TBD | TBD | TBD | TBD |
| Baseline | Gloss→Text | TBD | TBD | TBD | TBD |
| Modern | Text→Gloss | TBD | TBD | TBD | TBD |
| Modern | Gloss→Text | TBD | TBD | TBD | TBD |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kagozi/asl.git
cd asl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

**Train Baseline Model (Text → Gloss):**
```bash
python experiments/train_baseline.py \\
    --direction text2gloss \\
    --config config/baseline_config.yaml \\
    --gpu 0
```

**Train Modern Model (Gloss → Text):**
```bash
python experiments/train_modern.py \\
    --direction gloss2text \\
    --config config/modern_config.yaml \\
    --gpu 0
```

**Train Both Directions:**
```bash
python experiments/train_baseline.py --direction both
python experiments/train_modern.py --direction both
```

### Evaluation

```bash
# Evaluate a specific model
python evaluation/evaluator.py \\
    --checkpoint results/checkpoints/modern_text2gloss_best.pt \\
    --test-data data/processed/test.pkl

# Compare all models
python experiments/compare_models.py
```

## 📁 Project Structure

```
text-gloss-translation/
├── config/              # Configuration files
├── data/                # Data loading and preprocessing
├── models/              # Model architectures
├── training/            # Training utilities
├── evaluation/          # Evaluation metrics and scripts
├── utils/               # Helper functions
├── experiments/         # Training scripts
├── results/             # Model checkpoints and results
└── notebooks/           # Jupyter notebooks for analysis
```

## 🔧 Configuration

Edit `config/baseline_config.yaml` or `config/modern_config.yaml`:

```yaml
model:
  embedding_dim: 512
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  dropout: 0.1

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  warmup_steps: 4000
  gradient_accumulation_steps: 1
```

## 📊 Dataset

Using ASLG-PC12 (American Sign Language Gloss Parallel Corpus):
- **Training**: 82,710 sentence pairs
- **Validation**: 4,000 sentence pairs
- **Test**: 4,145 sentence pairs

The dataset is automatically downloaded from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("achrafothman/aslg_pc12")
```

## 🧪 Experiments

### Baseline vs Modern Architecture

Compare standard transformer with modern improvements:
```bash
python experiments/compare_models.py --experiment architecture
```

### Bidirectional Analysis

Analyze performance differences between text→gloss and gloss→text:
```bash
python experiments/compare_models.py --experiment bidirectional
```

### Ablation Studies

Test individual components:
```bash
python experiments/ablation.py --component rope  # Test without RoPE
python experiments/ablation.py --component gqa   # Test without GQA
```

## 📈 Monitoring Training

View training progress with tensorboard:
```bash
tensorboard --logdir results/logs
```

Or use the built-in plotting:
```bash
python utils/visualization.py --log-dir results/logs/baseline_text2gloss
```

## 🔬 Key Components

### Models

- **Baseline Transformer** (`models/baseline_transformer.py`): Standard transformer with sinusoidal positional encoding
- **Modern Transformer** (`models/modern_transformer.py`): Enhanced with RoPE, GQA, RMSNorm, SwiGLU

### Training

- **Warmup Scheduler** (`training/scheduler.py`): Noam learning rate schedule
- **Label Smoothing** (`training/loss.py`): Regularization technique
- **Mixed Precision** (`training/trainer.py`): Faster training with AMP

### Evaluation

- **BLEU Score**: Standard MT metric
- **METEOR**: Semantic similarity metric
- **chrF++**: Character-level F-score
- **ROUGE**: Recall-oriented metric

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{kagozi2025textgloss,
  title={Modern Transformer Architectures for Bidirectional Text-Gloss Translation},
  author={Alex Kagozi},
  year={2025},
  school={University of South Dakota}
}
```

## 📝 License

MIT License - see LICENSE file for details

## 🔮 Future Work

- [ ] Multi-dataset training (PHOENIX-2014T, CSL-Daily)
- [ ] Data augmentation (back-translation, paraphrasing)
- [ ] Attention visualization
- [ ] Human evaluation study
- [ ] Real-time inference API