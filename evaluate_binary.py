"""
Comprehensive evaluation for binary fire classifier.
Correctly handles 2-class output: [no_fire, fire]
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from perception_model import FirePerceptionCNN

print("="*70)
print("BINARY FIRE CLASSIFIER EVALUATION")
print("="*70)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FirePerceptionCNN(embedding_dim=128, pretrained=True)
checkpoint = torch.load('models/perception_cnn.pth', map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model state dict")

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_directory(image_dir, expected_class, class_name, max_images=100):
    """
    Evaluate images from a directory.
    
    Args:
        image_dir: Path to images
        expected_class: 0 for no_fire, 1 for fire
        class_name: Display name
        max_images: Max number of images to test
    """
    images = list(Path(image_dir).glob('*.jpg'))[:max_images]
    
    if not images:
        print(f"⚠️  No images in {image_dir}")
        return None
    
    correct = 0
    confidences = []
    
    print(f"\n{class_name} ({len(images)} images):")
    
    for img_path in tqdm(images, desc="  Evaluating"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, class_logits = model(img_tensor)
            
            # Binary classification: [no_fire, fire]
            probs = torch.softmax(class_logits, dim=1)[0]
            pred_class = torch.argmax(class_logits, dim=1).item()
            fire_conf = probs[1].item()  # Confidence that it's fire
            
            confidences.append(fire_conf)
            
            if pred_class == expected_class:
                correct += 1
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    accuracy = (correct / len(images)) * 100
    avg_conf = np.mean(confidences)
    
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Avg Fire Confidence: {avg_conf:.1%}")
    
    return {
        'accuracy': accuracy,
        'confidences': confidences,
        'total': len(images),
        'correct': correct
    }

# Evaluate on both classes
fire_results = evaluate_directory(
    'data/satellite_real/train/fire',
    expected_class=1,
    class_name="🔥 FIRE IMAGES"
)

nofire_results = evaluate_directory(
    'data/satellite_real/train/no_fire',
    expected_class=0,
    class_name="✓ NO-FIRE IMAGES"
)

# Final metrics
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

if fire_results and nofire_results:
    overall_acc = (fire_results['accuracy'] + nofire_results['accuracy']) / 2
    
    print(f"\n📊 Classification Performance:")
    print(f"  Fire Detection:     {fire_results['accuracy']:.1f}%")
    print(f"  No-Fire Detection:  {nofire_results['accuracy']:.1f}%")
    print(f"  Overall Accuracy:   {overall_acc:.1f}%")
    
    # Confidence analysis
    fire_conf = np.mean(fire_results['confidences'])
    nofire_conf = np.mean(nofire_results['confidences'])
    print(f"\n🎯 Confidence Analysis:")
    print(f"  Fire images → Fire conf:    {fire_conf:.1%}")
    print(f"  No-Fire images → Fire conf: {nofire_conf:.1%}")
    print(f"  Separation:                 {abs(fire_conf - nofire_conf):.1%}")
    
    # Calculate precision, recall, F1
    tp = fire_results['correct']
    fp = nofire_results['total'] - nofire_results['correct']
    fn = fire_results['total'] - fire_results['correct']
    tn = nofire_results['correct']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Advanced Metrics:")
    print(f"  Precision:  {precision*100:.1f}%")
    print(f"  Recall:     {recall*100:.1f}%")
    print(f"  F1-Score:   {f1*100:.1f}%")
    
    # Production readiness
    print(f"\n🚀 Production Readiness:")
    if overall_acc >= 85 and abs(fire_conf - nofire_conf) >= 0.3:
        print("  ✅ EXCELLENT: Ready for deployment")
    elif overall_acc >= 75 and abs(fire_conf - nofire_conf) >= 0.2:
        print("  ✓ GOOD: Minor improvements recommended")
    elif overall_acc >= 60:
        print("  ⚠️  FAIR: Needs more training")
    else:
        print("  ❌ POOR: Requires redesign or more data")

print("\n" + "="*70)
