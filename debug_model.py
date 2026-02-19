"""Quick test to see what the model is actually predicting"""
import torch
from perception_model import FirePerceptionCNN
from PIL import Image
from torchvision import transforms
import glob

print("Loading model...")
model = FirePerceptionCNN(embedding_dim=128, pretrained=True)
checkpoint = torch.load('models/perception_cnn.pth', map_location='cpu', weights_only=False)

# Check what's in the checkpoint
print(f"Checkpoint type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {checkpoint.keys()}")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            print(f"✓ Training achieved {checkpoint['val_acc']:.2f}% validation accuracy")
        if 'epoch' in checkpoint:
            print(f"✓ Saved at epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()

# Test on a few images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fire_imgs = glob.glob('data/satellite_real/train/fire/*.jpg')[:5]
print(f"\nTesting on {len(fire_imgs)} fire images:")

for img_path in fire_imgs:
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding, class_logits = model(img_tensor)
        
    probs = torch.softmax(class_logits[0], dim=0)
    print(f"  Logits: {class_logits}")
    print(f"  Probabilities: {probs}")
    print(f"  Fire confidence: {probs[1].item():.2%}")

print(f"\nTesting on 5 NO-FIRE images:")
nofire_imgs = glob.glob('data/satellite_real/train/no_fire/*.jpg')[:5]

for img_path in nofire_imgs:
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding, class_logits = model(img_tensor)
        
    probs = torch.softmax(class_logits[0], dim=0)
    print(f"  Logits: {class_logits}")
    print(f"  Probabilities: {probs}")
    print(f"  Fire confidence: {probs[1].item():.2%}")
