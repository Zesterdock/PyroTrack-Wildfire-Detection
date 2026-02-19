# 🔥 PyroTrack: Satellite Wildfire Detection

> Binary fire classifier achieving **87% accuracy** on satellite imagery using deep learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

PyroTrack is a satellite-based wildfire detection system that classifies images as containing fire or no fire. The system uses a **MobileNetV2** backbone with custom training strategies to achieve production-ready performance.

### Key Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **87.0%** |
| Fire Detection Rate | 80.0% |
| No-Fire Detection Rate | 94.0% |
| Precision | 93.0% |
| Recall | 80.0% |
| F1-Score | 86.0% |

✅ **Status:** Ready for deployment as triage/screening system

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Zesterdock/PyroTrack-Wildfire-Detection.git
cd PyroTrack-Wildfire-Detection

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the [Kaggle Wildfire Detection Dataset](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data) and organize as:

```
data/satellite_real/train/
├── fire/       # 15,750 images
└── no_fire/    # 14,500 images
```

### Train Model

```bash
python perception_model.py --data-source satellite \
                          --data-dir data/satellite_real \
                          --epochs 2 \
                          --batch-size 16
```

**Training time:** ~30 minutes on CPU

### Evaluate Model

```bash
# Comprehensive evaluation
python evaluate_binary.py

# Quick debug test
python debug_model.py
```

## 📊 Performance Details

### Confusion Matrix

|  | Predicted: Fire | Predicted: No-Fire |
|--|----------------|-------------------|
| **Actual: Fire** | 80 (TP) | 20 (FN) |
| **Actual: No-Fire** | 6 (FP) | 94 (TN) |

### Confidence Analysis

- **Fire images:** 58.5% avg confidence
- **No-fire images:** 11.4% avg confidence  
- **Separation:** 47.1% (excellent discrimination)

## 🏗️ Architecture

```
Input (128×128 RGB)
       ↓
MobileNetV2 (pretrained)
       ↓
Embedding Layer (128-dim)
       ↓
Binary Classifier
       ↓
Output [no_fire, fire]
```

### Training Strategy

**Key Innovation:** 10:1 class weighting prevents false positives

```python
class_weights = [10.0, 1.0]  # [no_fire, fire]
criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

**Why this works:**
- Without weighting: 99% false positives ❌
- With 10x weighting: 6% false positives ✅

### Optimal Training

- **Epochs:** 2 (prevents overfitting)
- **Batch size:** 16
- **Optimizer:** Adam (lr=1e-3)
- **Scheduler:** ReduceLROnPlateau

## 📁 Project Structure

```
PyroTrack-Wildfire-Detection/
├── perception_model.py       # Main training script
├── evaluate_binary.py         # Comprehensive evaluation
├── debug_model.py             # Quick prediction testing
├── wildfire_env.py            # POMDP wildfire simulator
├── cot_module.py              # Chain-of-Thought module
├── setup_satellite_easy.py    # Dataset download helper
├── requirements.txt           # Python dependencies
├── PROJECT_SUMMARY.md         # Detailed technical overview
└── models/
    └── README.md              # Model training instructions
```

## 🔬 Key Learnings

### 1. Class Weighting is Critical
Initial training produced 99% fire predictions. 10:1 weighting fixed this completely.

### 2. Less is More
| Epochs | Fire Detection | No-Fire Detection | Overall |
|--------|---------------|------------------|---------|
| 2 ✅ | 80% | 94% | **87%** |
| 35 ❌ | 88% | 19% | 53.5% |

More epochs = worse performance due to overfitting.

### 3. Architecture Simplification
- **Removed:** Heatmap prediction task (conflicting objectives)
- **Removed:** 3-class output (smoke class unnecessary)
- **Result:** 34% accuracy gain

## 🌍 Real-World Applications

### ✅ Suitable For:
- **Triage/Screening** - Rapid initial assessment of satellite imagery
- **Supplementary Monitoring** - Complement to existing systems
- **Alert Prioritization** - Focus resources on high-confidence detections
- **Research & Development** - Academic wildfire detection studies

### ⚠️ Limitations:
- **20% miss rate** - Not suitable as sole early warning system
- **Single sensor type** - Limited to optical satellite imagery
- **No temporal analysis** - Single-frame classification only

**Recommendation:** Combine with multi-sensor fusion and human verification for production deployment

## 📈 Dataset

**Source:** [brsdincer/wildfire-detection-image-data](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data)

- **Total images:** 30,250
- **Fire:** 15,750 images
- **No-fire:** 14,500 images
- **Type:** Satellite/aerial wildfire imagery
- **Split:** 80/20 train/validation

## 🛠️ Technical Stack

- **Framework:** PyTorch 2.0+
- **Architecture:** MobileNetV2 (ImageNet pretrained)
- **Training:** Stable-Baselines3 for RL components
- **Visualization:** Matplotlib, TensorBoard

## 📝 Citation

```bibtex
@software{pyrotrack2026,
  title = {PyroTrack: Satellite Wildfire Detection System},
  author = {Pranjal},
  year = {2026},
  url = {https://github.com/Zesterdock/PyroTrack-Wildfire-Detection},
  note = {87\% accuracy binary fire classifier}
}
```

## 🤝 Contributing

Contributions welcome! Potential improvements:

- [ ] Multi-sensor fusion (thermal + optical)
- [ ] Temporal analysis (video sequences)
- [ ] Smoke detection
- [ ] Transfer learning to new regions
- [ ] Real-time inference optimization

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- **Dataset:** brsdincer (Kaggle)
- **Framework:** PyTorch team
- **Pretrained backbone:** MobileNetV2 (ImageNet)

---

**Project Status:** ✅ Complete  
**Best Model:** 87% accuracy (2 epochs, 10x class weighting)  
**Last Updated:** February 2026
