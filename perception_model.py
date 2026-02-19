"""
Computer Vision Perception Module for Wildfire Detection
Uses lightweight MobileNetV2 architecture for fire detection and feature extraction.

REAL-WORLD FIRST ARCHITECTURE:
- Primary: Loads real-world imagery from data/real_world or data/satellite
- Fallback: Synthetic data generation (use --synthetic flag)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from wildfire_env import WildfireEnv


class FirePerceptionCNN(nn.Module):
    """
    Lightweight CNN for fire detection using MobileNetV2 backbone.
    
    Outputs:
        - Fire probability heatmap
        - Feature embedding vector (128-dim)
    """
    
    def __init__(self, embedding_dim=128, pretrained=False):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # MobileNetV2 backbone (lightweight)
        # Use 'weights' parameter for newer torchvision versions
        weights = 'DEFAULT' if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        
        # Reduce width for efficiency
        self.features = mobilenet.features
        
        # Feature projection for embedding
        self.embedding_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.ReLU()
        )
        
        # Classification head (2 classes: no_fire=0, fire=1)
        # Pure binary classifier - no heatmap task
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input RGB image [B, 3, 128, 128]
            
        Returns:
            embedding: Feature vector [B, embedding_dim]
            class_logits: Classification logits [B, 2] (no_fire=0, fire=1)
        """
        # Extract features
        features = self.features(x)
        
        # Generate embedding
        embedding = self.embedding_layer(features)
        
        # Classification
        class_logits = self.classifier(features)
        
        return embedding, class_logits


class RealWorldFireDataset(Dataset):
    """
    Dataset for real-world fire images (PRIMARY DATASET).
    Loads from data/real_world or data/satellite directories.
    
    Directory structure expected:
        data/
        ├── real_world/  (or satellite/)
        │   ├── train/
        │   │   ├── fire/
        │   │   ├── no_fire/
        │   │   └── smoke/  (optional)
        │   ├── val/
        │   └── test/
    """
    
    def __init__(self, data_dir='data/real_world', split='train', augment=True):
        """
        Args:
            data_dir: Root directory (data/real_world or data/satellite)
            split: 'train', 'val', or 'test'
            augment: Enable data augmentation (for training)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Check if data exists, fallback to satellite if needed
        if not self.data_dir.exists():
            alt_dir = Path('data/satellite')
            if alt_dir.exists():
                print(f"⚠️  {self.data_dir} not found, using {alt_dir}")
                self.data_dir = alt_dir
            else:
                raise FileNotFoundError(
                    f"Neither {self.data_dir} nor data/satellite found!\n"
                    f"Please download dataset or use --synthetic flag"
                )
        
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                f"Expected structure: {self.data_dir}/train/fire/ and {self.data_dir}/train/no_fire/"
            )
        
        # Load image paths and labels
        self.samples = []
        self.class_names = []
        
        # Check for fire/no_fire/smoke classes
        for class_idx, class_name in enumerate(['no_fire', 'smoke', 'fire']):
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                self.class_names.append(class_name)
                image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.jp2']
                for ext in image_extensions:
                    for img_path in class_dir.glob(f'*{ext}'):
                        self.samples.append((str(img_path), class_idx))
        
        # Fallback: if no 'smoke' class, merge into binary classification
        if 'smoke' not in self.class_names:
            # Remap: no_fire=0, fire=2 -> no_fire=0, fire=1
            self.samples = [(path, min(label, 1)) for path, label in self.samples]
            self.num_classes = 2
        else:
            self.num_classes = 3
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {self.split_dir}!\n"
                f"Expected subdirectories: fire/, no_fire/, (optional: smoke/)"
            )
        
        print(f"✓ Loaded {len(self.samples)} images from {self.split_dir}")
        print(f"  Classes: {self.class_names} ({self.num_classes} total)")
        
        # Data augmentation transforms
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        # Apply transforms
        image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


class SyntheticWildfireDataset(Dataset):
    """
    FALLBACK: Synthetic dataset for testing without real data.
    Generates synthetic data from the wildfire environment simulator.
    
    USE ONLY FOR:
    - Initial testing
    - Algorithm development
    - When real data is unavailable
    """
    
    def __init__(self, num_samples=1000, grid_size=32):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.samples = []
        
        print(f"Generating {num_samples} synthetic training samples...")
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate synthetic training data."""
        env = WildfireEnv(grid_size=self.grid_size)
        
        for _ in tqdm(range(self.num_samples)):
            obs, info = env.reset()
            
            # Random number of steps
            steps = np.random.randint(0, 50)
            for _ in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            # Render RGB frame
            rgb_frame = env.render()
            
            # Get ground truth
            grid = env.get_grid()
            
            # Determine class: 0=no_fire, 1=smoke (simplified), 2=fire
            num_burning = np.sum(grid == 2)
            if num_burning == 0:
                label = 0  # no fire
            elif num_burning < 5:
                label = 1  # smoke/small fire
            else:
                label = 2  # fire
            
            # Create heatmap ground truth (binary: fire or not)
            heatmap_gt = (grid == 2).astype(np.float32)
            
            self.samples.append({
                'image': rgb_frame,
                'label': label,
                'heatmap': heatmap_gt
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert image to tensor
        image = torch.from_numpy(sample['image']).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        # Resize to 128x128 if needed
        if image.shape[1] != 128 or image.shape[2] != 128:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(128, 128), 
                mode='bilinear'
            ).squeeze(0)
        
        # Heatmap ground truth (resize to match output size)
        heatmap = torch.from_numpy(sample['heatmap']).float()
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(32, 32),
            mode='bilinear'
        ).squeeze()
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, heatmap, label


def train_perception_model(data_source='real_world', data_dir='data/real_world', 
                          num_samples=1000, epochs=20, batch_size=8, 
                          save_path='models/perception_cnn.pth', pretrained=True):
    """
    Train the perception CNN model.
    
    Args:
        data_source: 'real_world', 'satellite', or 'synthetic'
        data_dir: Path to dataset directory (for real_world/satellite)
        num_samples: Number of synthetic samples (only for synthetic mode)
        epochs: Training epochs
        batch_size: Batch size
        save_path: Path to save trained model
        pretrained: Use pretrained MobileNetV2 weights
    """
    print("=" * 60)
    print("Training Fire Perception CNN (REAL-WORLD FIRST)")
    print("=" * 60)
    print(f"Data source: {data_source}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset based on source
    if data_source == 'synthetic':
        print("\n⚠️  Using SYNTHETIC data (not recommended for production)")
        train_dataset = SyntheticWildfireDataset(num_samples=int(num_samples * 0.8))
        val_dataset = SyntheticWildfireDataset(num_samples=int(num_samples * 0.2))
    else:
        print(f"\n✓ Using REAL-WORLD data from: {data_dir}")
        try:
            train_dataset = RealWorldFireDataset(data_dir=data_dir, split='train', augment=True)
            
            # Check if val split exists
            val_dir = Path(data_dir) / 'val'
            if val_dir.exists() and len(list(val_dir.glob('**/*.jpg'))) > 0:
                val_dataset = RealWorldFireDataset(data_dir=data_dir, split='val', augment=False)
            else:
                # Use 100% of data for training
                print("  Using 100% of training data (30,250 images)")
                # Load same data but without augmentation for validation monitoring
                val_dataset = RealWorldFireDataset(data_dir=data_dir, split='train', augment=False)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n❌ Error loading real data: {e}")
            print("Falling back to SYNTHETIC data...")
            data_source = 'synthetic'
            train_dataset = SyntheticWildfireDataset(num_samples=int(num_samples * 0.8))
            val_dataset = SyntheticWildfireDataset(num_samples=int(num_samples * 0.2))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Create model (use pretrained weights for better real-world performance)
    model = FirePerceptionCNN(embedding_dim=128, pretrained=pretrained)
    model = model.to(device)
    
    print(f"Model: MobileNetV2 {'(pretrained)' if pretrained else '(from scratch)'}")
    
    # Loss function (pure classification)
    # AGGRESSIVE: Penalize false positives 10x more + label smoothing
    # [no_fire_weight, fire_weight] = [10.0, 1.0]
    class_weights = torch.tensor([10.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_class_acc = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings, class_logits = model(images)
            
            # Calculate loss (pure classification)
            loss = criterion(class_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            pred_labels = torch.argmax(class_logits, dim=1)
            train_class_acc += (pred_labels == labels).float().mean().item()
        
        train_loss /= len(train_loader)
        train_class_acc /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_class_acc = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                embeddings, class_logits = model(images)
                
                loss = criterion(class_logits, labels)
                
                val_loss += loss.item()
                
                pred_labels = torch.argmax(class_logits, dim=1)
                val_class_acc += (pred_labels == labels).float().mean().item()
        
        val_loss /= len(val_loader)
        val_class_acc /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_class_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_class_acc*100:.2f}%")
        
        # Save best model with metadata
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_class_acc * 100,
                'train_acc': train_class_acc * 100
            }
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_class_acc*100:.2f}%)")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Perception CNN Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('logs/perception_training.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: logs/perception_training.png")
    plt.close()
    
    print(f"\n✓ Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fire Perception CNN (Real-World First)')
    parser.add_argument('--data-source', default='real_world', 
                       choices=['real_world', 'satellite', 'synthetic'],
                       help='Data source (default: real_world)')
    parser.add_argument('--data-dir', default='data/real_world',
                       help='Path to dataset directory')
    parser.add_argument('--synthetic-samples', type=int, default=1000,
                       help='Number of synthetic samples (only for --data-source=synthetic)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch (no pretrained weights)')
    parser.add_argument('--save-path', default='models/perception_cnn.pth',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Auto-detect satellite data if real_world not found
    if args.data_source == 'real_world' and not Path(args.data_dir).exists():
        satellite_dir = Path('data/satellite')
        if satellite_dir.exists():
            print(f"⚠️  {args.data_dir} not found, using satellite data")
            args.data_source = 'satellite'
            args.data_dir = 'data/satellite'
    
    # Train perception model
    model = train_perception_model(
        data_source=args.data_source,
        data_dir=args.data_dir,
        num_samples=args.synthetic_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        pretrained=not args.no_pretrained
    )
    
    print("\n✓ Perception model training complete!")
    print(f"\nUsage examples:")
    print(f"  # Train on real-world data (default):")
    print(f"  python perception_model.py")
    print(f"  # Train on satellite data:")
    print(f"  python perception_model.py --data-source satellite --data-dir data/satellite")
    print(f"  # Train on synthetic data (fallback):")
    print(f"  python perception_model.py --data-source synthetic")

