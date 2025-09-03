"""
Basic training script for Text-to-Doodle model
"""
import torch
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader

from model import create_model, TextToSketchTrainer, QuickDrawDataset

def load_quickdraw_data(data_dir="./data", max_samples_per_category=1000):
    """Load QuickDraw data from numpy files"""
    
    # Get all .npy files in data directory
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    categories = [f[:-4] for f in npy_files]  # Remove .npy extension
    
    print(f"Found {len(categories)} categories: {categories[:5]}...")
    
    all_sketches = []
    all_labels = []
    
    for category in categories:
        filepath = os.path.join(data_dir, f"{category}.npy")
        
        try:
            # Load and preprocess
            data = np.load(filepath)
            data = data.astype(np.float32) / 255.0  # Normalize to [0,1]
            data = data.reshape(-1, 28, 28)  # Ensure correct shape
            
            # Limit samples
            if len(data) > max_samples_per_category:
                data = data[:max_samples_per_category]
            
            all_sketches.append(data)
            all_labels.extend([category] * len(data))
            
            print(f"Loaded {len(data)} samples for {category}")
            
        except Exception as e:
            print(f"Error loading {category}: {e}")
            continue
    
    # Combine all sketches
    all_sketches = np.concatenate(all_sketches, axis=0)
    
    print(f"\nTotal dataset: {len(all_sketches)} sketches, {len(set(all_labels))} categories")
    
    return all_sketches, all_labels

def train_basic():
    """Basic training function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 5
    max_samples = 500  # Small for basic training
    
    print(f"Device: {device}")
    print(f"Starting basic training...")
    
    # Load data
    print("Loading QuickDraw data...")
    sketches, labels = load_quickdraw_data(max_samples_per_category=max_samples)
    
    # Create dataset and dataloader
    dataset = QuickDrawDataset(sketches, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Dataloader has {len(dataloader)} batches")
    
    # Create model
    print("Creating VAE model...")
    model = create_model('vae')
    trainer = TextToSketchTrainer(model, device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (target_sketches, text_prompts, labels) in enumerate(dataloader):
            target_sketches = target_sketches.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            generated_sketches, mu, logvar, text_embeddings = model(text_prompts)
            
            # Calculate loss
            loss, recon_loss, kl_loss, clip_loss = trainer.vae_loss(
                generated_sketches, target_sketches, mu, logvar, text_prompts
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx:3d}: Loss={loss.item():.4f}, "
                      f"Recon={recon_loss.item():.4f}, "
                      f"KL={kl_loss.item():.4f}, "
                      f"CLIP={clip_loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    print("\nBasic training completed!")
    print("Model is ready for more extensive training or evaluation.")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        test_prompts = ["a sketch of a cat", "drawing of a dog", "a doodle of a bird"]
        generated, _, _, _ = model(test_prompts)
        print(f"Generated {len(test_prompts)} sketches with shape: {generated.shape}")
    
    return model, trainer

if __name__ == "__main__":
    # Run basic training
    try:
        model, trainer = train_basic()
        print("Training script completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
