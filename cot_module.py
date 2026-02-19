"""
Visual Chain-of-Thought (CoT) Module
Generates structured reasoning steps from visual features to guide UAV navigation.

REAL-WORLD FIRST ARCHITECTURE:
- Primary: Extracts features from real-world imagery
- Generates reasoning based on actual fire characteristics
- Fallback: Synthetic reasoning (use --synthetic flag)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from PIL import Image

from wildfire_env import WildfireEnv
from perception_model import FirePerceptionCNN, RealWorldFireDataset


class VisualCoTModule(nn.Module):
    """
    Visual Chain-of-Thought reasoning module.
    
    Takes CNN features and generates a sequence of reasoning steps.
    Uses LSTM decoder with attention.
    """
    
    def __init__(self, feature_dim=128, hidden_dim=128, vocab_size=50, max_length=4, cot_embedding_dim=64):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.cot_embedding_dim = cot_embedding_dim
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # CoT embedding projection (for RL integration)
        self.cot_embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim * max_length, 128),
            nn.ReLU(),
            nn.Linear(128, cot_embedding_dim)
        )
        
        # Special tokens
        self.START_TOKEN = 0
        self.END_TOKEN = 1
        self.PAD_TOKEN = 2
    
    def forward(self, features, target_tokens=None, teacher_forcing_ratio=0.5):
        """
        Args:
            features: CNN feature embeddings [B, feature_dim]
            target_tokens: Ground truth tokens [B, max_length] (for training)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            logits: Token logits [B, max_length, vocab_size]
            cot_embedding: Fixed-size embedding [B, cot_embedding_dim]
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Project features to hidden dimension
        features = self.feature_proj(features)
        
        # Initialize LSTM hidden state with features
        h0 = features.unsqueeze(0).repeat(2, 1, 1)  # [num_layers, B, hidden_dim]
        c0 = torch.zeros_like(h0)
        
        # Start with START token
        current_token = torch.full(
            (batch_size,), 
            self.START_TOKEN, 
            dtype=torch.long, 
            device=device
        )
        
        outputs = []
        hidden_states = []
        hidden = (h0, c0)
        
        for t in range(self.max_length):
            # Embed current token
            token_embed = self.token_embedding(current_token).unsqueeze(1)  # [B, 1, hidden_dim]
            
            # LSTM step
            output, hidden = self.lstm(token_embed, hidden)
            hidden_states.append(output.squeeze(1))
            
            # Project to vocabulary
            logits = self.output_proj(output.squeeze(1))  # [B, vocab_size]
            outputs.append(logits)
            
            # Teacher forcing or use prediction
            if target_tokens is not None and np.random.rand() < teacher_forcing_ratio:
                current_token = target_tokens[:, t]
            else:
                current_token = torch.argmax(logits, dim=-1)
        
        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # [B, max_length, vocab_size]
        
        # Create CoT embedding from hidden states
        hidden_concat = torch.cat(hidden_states, dim=1)  # [B, max_length * hidden_dim]
        cot_embedding = self.cot_embedding_proj(hidden_concat)  # [B, cot_embedding_dim]
        
        return logits, cot_embedding
    
    def generate_reasoning(self, features, vocab):
        """
        Generate reasoning text from features.
        
        Args:
            features: CNN features [B, feature_dim]
            vocab: Vocabulary mapping {token_id: word}
            
        Returns:
            reasoning_texts: List of reasoning sequences
        """
        self.eval()
        with torch.no_grad():
            logits, cot_embedding = self.forward(features, target_tokens=None)
            tokens = torch.argmax(logits, dim=-1)  # [B, max_length]
        
        reasoning_texts = []
        for token_seq in tokens:
            words = []
            for token_id in token_seq:
                token_id = token_id.item()
                if token_id == self.END_TOKEN:
                    break
                if token_id not in [self.START_TOKEN, self.PAD_TOKEN]:
                    word = vocab.get(token_id, f"<UNK_{token_id}>")
                    words.append(word)
            reasoning_texts.append(" ".join(words))
        
        return reasoning_texts


class ReasoningVocabulary:
    """
    Simple vocabulary for reasoning tokens.
    """
    
    def __init__(self):
        # Define reasoning vocabulary
        self.words = [
            "<START>", "<END>", "<PAD>",
            "smoke", "detected", "fire", "region", "northeast", "northwest",
            "southeast", "southwest", "north", "south", "east", "west",
            "center", "high", "low", "intensity", "hotspot", "visible",
            "spreading", "wind", "pushing", "recommend", "moving", "towards",
            "away", "monitoring", "area", "cells", "burning", "front",
            "small", "large", "active", "stable", "increasing", "decreasing",
            "priority", "critical", "caution", "safe", "danger", "alert",
            "multiple", "single", "cluster", "isolated"
        ]
        
        self.word_to_id = {word: idx for idx, word in enumerate(self.words)}
        self.id_to_word = {idx: word for idx, word in enumerate(self.words)}
        
        self.vocab_size = len(self.words)
    
    def encode(self, text):
        """Encode text to token IDs."""
        words = text.lower().split()
        return [self.word_to_id.get(word, 2) for word in words]  # 2 = PAD
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        words = [self.id_to_word.get(tok, "<UNK>") for tok in token_ids]
        # Remove special tokens
        words = [w for w in words if w not in ["<START>", "<END>", "<PAD>"]]
        return " ".join(words)


class RealWorldCoTDataset(Dataset):
    """
    PRIMARY: CoT dataset from real-world imagery.
    Extracts features from real fire images and generates reasoning.
    """
    
    def __init__(self, data_dir='data/real_world', split='train', perception_model=None):
        """
        Args:
            data_dir: Path to real-world dataset
            split: 'train', 'val', or 'test'
            perception_model: Trained FirePerceptionCNN for feature extraction
        """
        self.data_dir = data_dir
        self.split = split
        self.vocab = ReasoningVocabulary()
        self.perception_model = perception_model
        
        # Load real-world dataset
        try:
            self.image_dataset = RealWorldFireDataset(
                data_dir=data_dir,
                split=split,
                augment=(split == 'train')
            )
            print(f"✓ Loaded {len(self.image_dataset)} real images for CoT training")
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load real-world data for CoT: {e}\n"
                f"Use --data-source synthetic as fallback"
            )
        
        # Setup perception model for feature extraction
        if self.perception_model is None:
            print("⚠️  No perception model provided, using random features")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.perception_model = self.perception_model.to(self.device)
            self.perception_model.eval()
    
    def _generate_reasoning_from_label(self, label, fire_prob=None):
        """
        Generate reasoning based on fire classification and probability.
        
        Args:
            label: 0=no_fire, 1=smoke, 2=fire (or 0/1 for binary)
            fire_prob: Optional fire probability from model
        """
        # Reasoning templates based on actual fire characteristics
        if label == 0:  # No fire
            templates = [
                "safe area monitoring",
                "no fire detected",
                "routine surveillance",
                "clear conditions"
            ]
        elif label == 1:  # Smoke (if 3-class)
            templates = [
                "smoke detected investigate",
                "potential fire area",
                "recommend closer inspection",
                "monitor smoke source"
            ]
        else:  # Fire (label == 2 or 1 in binary)
            if fire_prob is not None and fire_prob > 0.8:
                templates = [
                    "fire confirmed critical",
                    "high intensity hotspot",
                    "immediate response required",
                    "priority monitoring area"
                ]
            else:
                templates = [
                    "fire detected moderate",
                    "hotspot identified",
                    "recommend investigation",
                    "monitor fire spread"
                ]
        
        # Select random template for variety
        reasoning = np.random.choice(templates)
        return reasoning
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Get image and label from real dataset
        image, heatmap, label = self.image_dataset[idx]
        
        # Extract features using perception model
        if self.perception_model is not None:
            with torch.no_grad():
                image_batch = image.unsqueeze(0).to(self.device)
                _, features, class_logits = self.perception_model(image_batch)
                features = features.cpu().numpy()[0]
                
                # Get fire probability
                fire_probs = F.softmax(class_logits, dim=1).cpu().numpy()[0]
                fire_prob = fire_probs[-1] if len(fire_probs) > 1 else fire_probs[0]
        else:
            features = np.random.randn(128)
            fire_prob = None
        
        # Generate reasoning based on label
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        reasoning_text = self._generate_reasoning_from_label(label_val, fire_prob)
        
        # Encode reasoning
        reasoning_tokens = self.vocab.encode(reasoning_text)
        
        # Pad to max length (4 tokens)
        if len(reasoning_tokens) < 4:
            reasoning_tokens += [self.vocab.word_to_id["<PAD>"]] * (4 - len(reasoning_tokens))
        else:
            reasoning_tokens = reasoning_tokens[:4]
        
        features = torch.from_numpy(features).float()
        tokens = torch.tensor(reasoning_tokens, dtype=torch.long)
        
        return features, tokens


class SyntheticCoTDataset(Dataset):
    """
    FALLBACK: Synthetic CoT dataset for testing without real data.
    Generates reasoning from environment simulation.
    """
    
    def __init__(self, num_samples=1000, perception_model=None):
        self.num_samples = num_samples
        self.perception_model = perception_model
        self.vocab = ReasoningVocabulary()
        self.samples = []
        
        print(f"Generating {num_samples} CoT training samples...")
        self._generate_samples()
    
    def _generate_reasoning_from_state(self, grid, uav_pos):
        """Generate reasoning text from grid state."""
        rows, cols = grid.shape
        burning_cells = np.argwhere(grid == 2)
        
        if len(burning_cells) == 0:
            return "safe area monitoring"
        
        # Analyze fire location relative to UAV
        fire_center = burning_cells.mean(axis=0)
        uav_array = np.array(uav_pos)
        
        # Direction from UAV to fire
        diff = fire_center - uav_array
        
        directions = []
        if diff[0] < -5:
            directions.append("north")
        elif diff[0] > 5:
            directions.append("south")
        
        if diff[1] < -5:
            directions.append("west")
        elif diff[1] > 5:
            directions.append("east")
        
        direction = "".join(directions) if directions else "center"
        
        # Fire intensity
        num_burning = len(burning_cells)
        if num_burning < 5:
            intensity = "small"
        elif num_burning < 15:
            intensity = "high"
        else:
            intensity = "critical"
        
        # Generate reasoning (max 4 tokens)
        reasoning_templates = [
            f"fire detected {direction}",
            f"{intensity} intensity hotspot",
            f"recommend moving towards",
            f"priority monitoring area"
        ]
        
        # Select based on state
        if num_burning < 3:
            reasoning = reasoning_templates[0]
        elif num_burning < 10:
            reasoning = reasoning_templates[1]
        else:
            reasoning = reasoning_templates[2]
        
        return reasoning
    
    def _generate_samples(self):
        """Generate synthetic training data."""
        env = WildfireEnv(grid_size=32)
        
        # Load perception model if provided
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.perception_model is not None:
            self.perception_model = self.perception_model.to(device)
            self.perception_model.eval()
        
        for _ in tqdm(range(self.num_samples)):
            obs, info = env.reset()
            
            # Random steps
            steps = np.random.randint(0, 50)
            for _ in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            # Get RGB frame
            rgb_frame = env.render()
            
            # Get features from perception model
            if self.perception_model is not None:
                with torch.no_grad():
                    image = torch.from_numpy(rgb_frame).float() / 255.0
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    if image.shape[-2:] != (128, 128):
                        image = F.interpolate(image, size=(128, 128), mode='bilinear')
                    
                    _, features, _ = self.perception_model(image)
                    features = features.cpu().numpy()[0]
            else:
                features = np.random.randn(128)
            
            # Generate reasoning
            grid = env.get_grid()
            reasoning_text = self._generate_reasoning_from_state(grid, env.uav_pos)
            
            # Encode reasoning
            reasoning_tokens = self.vocab.encode(reasoning_text)
            
            # Pad to max length
            if len(reasoning_tokens) < 4:
                reasoning_tokens += [self.vocab.word_to_id["<PAD>"]] * (4 - len(reasoning_tokens))
            else:
                reasoning_tokens = reasoning_tokens[:4]
            
            self.samples.append({
                'features': features,
                'tokens': reasoning_tokens,
                'text': reasoning_text
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = torch.from_numpy(sample['features']).float()
        tokens = torch.tensor(sample['tokens'], dtype=torch.long)
        
        return features, tokens


def train_cot_module(perception_model_path='models/perception_cnn.pth',
                     data_source='real_world', data_dir='data/real_world',
                     num_samples=1000, epochs=30, batch_size=16,
                     save_path='models/cot_module.pth'):
    """
    Train the Visual CoT module (REAL-WORLD FIRST).
    
    Args:
        perception_model_path: Path to trained perception model
        data_source: 'real_world', 'satellite', or 'synthetic'
        data_dir: Path to dataset directory (for real_world/satellite)
        num_samples: Number of synthetic samples (only for synthetic mode)
        epochs: Training epochs
        batch_size: Batch size
        save_path: Path to save trained model
    """
    print("=" * 60)
    print("Training Visual Chain-of-Thought Module (REAL-WORLD FIRST)")
    print("=" * 60)
    print(f"Data source: {data_source}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load perception model for feature extraction
    perception_model = FirePerceptionCNN(embedding_dim=128, pretrained=False)
    if os.path.exists(perception_model_path):
        perception_model.load_state_dict(torch.load(perception_model_path, map_location=device, weights_only=True))
        print(f"✓ Loaded perception model from: {perception_model_path}")
    else:
        print("⚠️  Perception model not found. Using random features.")
        perception_model = None
    
    # Create dataset based on source
    if data_source == 'synthetic':
        print("\n⚠️  Using SYNTHETIC data (not recommended for production)")
        dataset = SyntheticCoTDataset(num_samples=num_samples, perception_model=perception_model)
    else:
        print(f"\n✓ Using REAL-WORLD data from: {data_dir}")
        try:
            dataset = RealWorldCoTDataset(
                data_dir=data_dir,
                split='train',
                perception_model=perception_model
            )
        except RuntimeError as e:
            print(f"\n❌ Error loading real data: {e}")
            print("Falling back to SYNTHETIC data...")
            data_source = 'synthetic'
            dataset = SyntheticCoTDataset(num_samples=num_samples, perception_model=perception_model)
    
    vocab = dataset.vocab
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create CoT model
    model = VisualCoTModule(
        feature_dim=128,
        hidden_dim=128,
        vocab_size=vocab.vocab_size,
        max_length=4,
        cot_embedding_dim=64
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word_to_id["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for features, tokens in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            tokens = tokens.to(device)
            
            # Forward pass
            logits, cot_embedding = model(features, target_tokens=tokens, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, vocab.vocab_size), tokens.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, tokens in val_loader:
                features = features.to(device)
                tokens = tokens.to(device)
                
                logits, cot_embedding = model(features, target_tokens=tokens, teacher_forcing_ratio=0.0)
                loss = criterion(logits.reshape(-1, vocab.vocab_size), tokens.reshape(-1))
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab
            }, save_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CoT Module Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('logs/cot_training.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: logs/cot_training.png")
    plt.close()
    
    print(f"\n✓ Model saved to: {save_path}")
    
    return model, vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Visual CoT Module (Real-World First)')
    parser.add_argument('--perception-model', default='models/perception_cnn.pth',
                       help='Path to trained perception model')
    parser.add_argument('--data-source', default='real_world',
                       choices=['real_world', 'satellite', 'synthetic'],
                       help='Data source (default: real_world)')
    parser.add_argument('--data-dir', default='data/real_world',
                       help='Path to dataset directory')
    parser.add_argument('--synthetic-samples', type=int, default=1000,
                       help='Number of synthetic samples (only for --data-source=synthetic)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--save-path', default='models/cot_module.pth',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Auto-detect satellite data if real_world not found
    if args.data_source == 'real_world' and not Path(args.data_dir).exists():
        satellite_dir = Path('data/satellite')
        if satellite_dir.exists():
            print(f"⚠️  {args.data_dir} not found, using satellite data")
            args.data_source = 'satellite'
            args.data_dir = 'data/satellite'
    
    # Train CoT module
    model, vocab = train_cot_module(
        perception_model_path=args.perception_model,
        data_source=args.data_source,
        data_dir=args.data_dir,
        num_samples=args.synthetic_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    print("\n✓ Visual CoT module training complete!")
    print(f"\nUsage examples:")
    print(f"  # Train on real-world data (default):")
    print(f"  python cot_module.py --perception-model models/perception_cnn.pth")
    print(f"  # Train on satellite data:")
    print(f"  python cot_module.py --data-source satellite --data-dir data/satellite")
    print(f"  # Train on synthetic data (fallback):")
    print(f"  python cot_module.py --data-source synthetic")

