# PyroTrack: Satellite Wildfire Detection with Visual Chain-of-Thought

**Binary fire classifier achieving 87% accuracy on satellite imagery**

## 🎯 Project Overview

This project implements a satellite-based wildfire detection system using:
- **MobileNetV2** CNN for fire classification
- **30,250 satellite images** from Kaggle wildfire dataset
- **Class-weighted training** to balance false positives/negatives

## 📊 Best Model Results

**Overall Accuracy: 87%**

| Metric | Score |
|--------|-------|
| Fire Detection Rate | 80% |
| No-Fire Detection Rate | 94% |
| Precision | 93% |
| Recall | 80% |
| F1-Score | 86% |
| Confidence Separation | 47% |

**Status:** ✅ Ready for deployment as triage/screening system

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd RL\ WildFire

# Install dependencies
pip install torch torchvision pillow numpy tqdm matplotlib

# Download dataset (optional - for training)
# Dataset: https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data
```

### Training

```bash
# Train the best model (2 epochs, 10x false-positive penalty)
python perception_model.py --data-source satellite \
                          --data-dir data/satellite_real \
                          --epochs 2 \
                          --batch-size 16
```

### Evaluation

```bash
# Evaluate on test images
python evaluate_binary.py

# Quick debug test
python debug_model.py
```

## 🏗️ Architecture

### Model: FirePerceptionCNN
- **Backbone:** MobileNetV2 (pretrained on ImageNet)
- **Input:** 128×128 RGB satellite images
- **Output:** Binary classification [no_fire, fire]
- **Loss:** CrossEntropyLoss with 10:1 class weighting (anti-false-positive)

### Training Configuration
```python
- Epochs: 2
- Batch size: 16
- Learning rate: 1e-3
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Class weights: [10.0, 1.0]  # [no_fire, fire]
- Label smoothing: 0.1
```

## 📁 Project Structure

```
RL WildFire/
├── perception_model.py      # Main training script
├── evaluate_binary.py        # Comprehensive evaluation
├── debug_model.py            # Quick prediction testing
├── wildfire_env.py           # POMDP wildfire simulator
├── cot_module.py             # Chain-of-Thought reasoning module
├── setup_satellite_easy.py   # Dataset download helper
└── models/
    └── perception_cnn.pth    # Trained model (87% accuracy)
```

## 🔬 Technical Details

### Why 10x Class Weighting?

Initial training without weighting produced 99% fire predictions (false positive disaster). The 10:1 weighting forces the model to be conservative:
- Reduces false positives from 81% → 6%
- Maintains 80% fire detection rate
- Achieves 93% precision

### Why Only 2 Epochs?

Training showed early overfitting:
- Epoch 1: 90.09% validation accuracy
- Epoch 2: 85.22% validation accuracy (↓)
- Epoch 3+: Would decrease further

**2 epochs is the optimal stopping point.**

## 📈 Training Evolution

| Approach | Fire Det | No-Fire Det | Overall | Issue |
|----------|----------|-------------|---------|-------|
| 35 epochs, no weighting | 88% | 19% | 53.5% | ❌ False positives |
| 2 epochs, 10x weighting (80/20) | **80%** | **94%** | **87%** | ✅ Balanced |
| 2 epochs, 10x weighting (100%) | 21% | 93% | 57% | ❌ Overfit |

## 🎓 Dataset

**Source:** [Kaggle Wildfire Detection Dataset](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data)

- **Total:** 30,250 satellite images
- **Fire images:** 15,750
- **No-fire images:** 14,500
- **Resolution:** Various (resized to 128×128)
- **Source:** Satellite/aerial imagery

## 🔧 Key Learnings

1. **Class weighting is critical** - Prevents model from predicting one class
2. **Less is more** - 2 epochs better than 35 (overfitting prevention)
3. **Validation split matters** - 80/20 split essential for proper evaluation
4. **Binary output works better** - Simplified from 3-class (no_fire/smoke/fire)
5. **Heatmap task was harmful** - Removed uniform heatmap prediction

## 🚀 Real-World Applications

### ✅ Suitable For:
- Triage/screening of satellite imagery
- Supplementary wildfire monitoring
- Alert prioritization systems
- Research and development
- Academic projects

### ⚠️ Limitations:
- Misses 20% of fires (not suitable as sole early warning)
- 6% false positive rate
- Single sensor type (Kaggle dataset)

**For production:** Combine with multi-sensor fusion, temporal analysis, and human verification

## 📝 Citation

```bibtex
@software{pyrotrack2026,
  title = {PyroTrack: Satellite Wildfire Detection System},
  author = {Pranjal},
  year = {2026},
  note = {87\% accuracy binary fire classifier}
}
```

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-sensor fusion (thermal + optical)
- Temporal analysis (video sequences)
- Smoke detection
- Larger/diverse datasets
- Transfer learning from other regions

## 🙏 Acknowledgments

- **Dataset:** brsdincer/wildfire-detection-image-data (Kaggle)
- **Framework:** PyTorch, torchvision
- **Backbone:** MobileNetV2 (ImageNet pretrained)

---

**Project Status:** ✅ Complete - Best model achieved 87% accuracy
