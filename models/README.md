# Models Directory

This directory contains trained model checkpoints.

## Best Model (Not included in repo due to size)

**File:** `perception_cnn.pth` (12MB)
**Performance:** 87% accuracy
**Download:** Train using `perception_model.py --epochs 2 --batch-size 16`

## Training Instructions

```bash
python perception_model.py --data-source satellite \
                          --data-dir data/satellite_real \
                          --epochs 2 \
                          --batch-size 16
```

This will generate:
- `perception_cnn.pth` - Best model checkpoint (87% accuracy)
- Training takes ~30 minutes on CPU
