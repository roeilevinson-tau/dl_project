import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import math

class CLIPTextEncoder(nn.Module):
    """CLIP-based text encoder for processing natural language prompts"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.text_encoder.config.hidden_size
        
    def forward(self, text_prompts):
        """
        Args:
            text_prompts: List of text strings
        Returns:
            text_embeddings: Tensor of shape (batch_size, embed_dim)
        """
        # Tokenize text
        inputs = self.tokenizer(
            text_prompts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=77
        )
        
        # Move inputs to the same device as the model
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).pooler_output
            
        return text_embeddings


class VAEEncoder(nn.Module):
    """Encoder for VAE that maps text embeddings to latent space"""
    
    def __init__(self, text_embed_dim=512, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(text_embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
    def forward(self, text_embeddings):
        h = F.relu(self.fc1(text_embeddings))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder for VAE that generates 28x28 sketches from latent code"""
    
    def __init__(self, latent_dim=128, output_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project latent to feature map
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        
        # Transposed convolutions for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),          # 28x28 -> 28x28
            nn.Sigmoid()  # Output binary images
        )
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 7, 7)
        output = self.deconv_layers(h)
        return output


class VAEBaseline(nn.Module):
    """VAE baseline model for text-to-sketch generation"""
    
    def __init__(self, text_embed_dim=512, latent_dim=128):
        super().__init__()
        self.text_encoder = CLIPTextEncoder()
        self.encoder = VAEEncoder(text_embed_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, text_prompts):
        # Get text embeddings
        text_embeddings = self.text_encoder(text_prompts)
        
        # Encode to latent space
        mu, logvar = self.encoder(text_embeddings)
        z = self.reparameterize(mu, logvar)
        
        # Decode to image
        generated_sketches = self.decoder(z)
        
        return generated_sketches, mu, logvar, text_embeddings


class DiffusionModel(nn.Module):
    """Simple diffusion model for conditional sketch generation"""
    
    def __init__(self, text_embed_dim=512, image_channels=1, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.text_encoder = CLIPTextEncoder()
        
        # U-Net like architecture for denoising
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.text_proj = nn.Linear(text_embed_dim, 64)
        
        # Simple CNN for denoising
        self.conv_layers = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, 3, padding=1),  # +1 for timestep
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_channels, 3, padding=1)
        )
        
    def forward(self, x, t, text_prompts):
        """
        Args:
            x: Noisy images (batch_size, 1, 28, 28)
            t: Timesteps (batch_size,)
            text_prompts: List of text strings
        """
        # Get text embeddings
        text_embeddings = self.text_encoder(text_prompts)
        text_features = self.text_proj(text_embeddings)  # (batch_size, 64)
        
        # Time embedding
        t_normalized = t.float().unsqueeze(1) / self.timesteps
        time_features = self.time_embed(t_normalized)  # (batch_size, 64)
        
        # Combine text and time features
        combined_features = text_features + time_features  # (batch_size, 64)
        
        # Broadcast to image dimensions
        batch_size = x.shape[0]
        feature_map = combined_features.view(batch_size, 64, 1, 1).expand(-1, -1, 28, 28)
        
        # Use first channel as conditioning
        conditioning = feature_map[:, :1, :, :]  # (batch_size, 1, 28, 28)
        
        # Concatenate input with conditioning
        conditioned_input = torch.cat([x, conditioning], dim=1)
        
        # Predict noise
        noise_pred = self.conv_layers(conditioned_input)
        
        return noise_pred


class CLIPGuidedLoss(nn.Module):
    """CLIP-guided loss for semantic consistency"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPVisionModel
        
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextEncoder()
        
        # Freeze CLIP parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Get embedding dimensions
        vision_dim = self.vision_encoder.config.hidden_size  # 768 for ViT-B/32
        text_dim = self.text_encoder.embed_dim  # 512 for CLIP text
        
        # Add projection layer to match dimensions
        self.vision_projection = nn.Linear(vision_dim, text_dim)
            
    def forward(self, generated_images, text_prompts):
        """
        Args:
            generated_images: Generated sketches (batch_size, 1, 28, 28)
            text_prompts: List of text strings
        Returns:
            CLIP similarity loss
        """
        # Convert grayscale to RGB for CLIP vision encoder
        if generated_images.shape[1] == 1:
            generated_images_rgb = generated_images.repeat(1, 3, 1, 1)
        else:
            generated_images_rgb = generated_images
            
        # Resize to 224x224 for CLIP
        generated_images_resized = F.interpolate(
            generated_images_rgb, 
            size=(224, 224), 
            mode='bilinear'
        )
        
        # Get image embeddings
        with torch.no_grad():
            # Ensure images are on the same device as vision encoder
            device = next(self.vision_encoder.parameters()).device
            generated_images_resized = generated_images_resized.to(device)
            raw_image_embeddings = self.vision_encoder(generated_images_resized).pooler_output
            
        # Project image embeddings to match text embedding dimension
        image_embeddings = self.vision_projection(raw_image_embeddings)
            
        # Get text embeddings
        text_embeddings = self.text_encoder(text_prompts)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(image_embeddings * text_embeddings, dim=-1)
        
        # Return negative similarity as loss (we want to maximize similarity)
        return -similarity.mean()


class QuickDrawDataset(Dataset):
    """Dataset class for QuickDraw sketches with text prompts"""
    
    def __init__(self, sketches, labels, text_templates=None):
        """
        Args:
            sketches: Numpy array of shape (N, 28, 28) containing binary sketches
            labels: List of object category labels
            text_templates: List of text templates for generating prompts
        """
        self.sketches = sketches
        self.labels = labels
        
        if text_templates is None:
            self.text_templates = [
                "a sketch of a {}",
                "drawing of a {}",
                "a doodle of a {}",
                "a simple drawing of a {}",
                "a black and white sketch of a {}"
            ]
        else:
            self.text_templates = text_templates
            
    def __len__(self):
        return len(self.sketches)
    
    def __getitem__(self, idx):
        sketch = torch.FloatTensor(self.sketches[idx]).unsqueeze(0)  # Add channel dimension
        label = self.labels[idx]
        
        # Generate text prompt using random template
        template = np.random.choice(self.text_templates)
        text_prompt = template.format(label)
        
        return sketch, text_prompt, label


class EvaluationMetrics:
    """Evaluation metrics for text-to-sketch generation"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.clip_loss = CLIPGuidedLoss().to(device)
        
    def clip_similarity_score(self, generated_sketches, text_prompts):
        """
        Compute CLIP similarity between generated sketches and text prompts
        Returns similarity score (higher is better)
        """
        # Use negative of CLIP loss to get positive similarity
        return -self.clip_loss(generated_sketches, text_prompts).item()
    
    def compute_fid_features(self, images):
        """
        Compute features for FID calculation using a simple CNN
        In practice, you'd use InceptionV3 features
        """
        # Simple feature extractor (placeholder for InceptionV3)
        feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(self.device)
        
        with torch.no_grad():
            features = feature_extractor(images)
        return features.cpu().numpy()
    
    def calculate_fid(self, real_images, generated_images):
        """
        Calculate Frechet Inception Distance (FID) score
        Lower is better
        """
        # Get features
        real_features = self.compute_fid_features(real_images)
        gen_features = self.compute_fid_features(generated_images)
        
        # Calculate means and covariances
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_gen
        covmean = np.sqrt(sigma_real @ sigma_gen)
        
        # Handle numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid
    
    def evaluate_model(self, model, dataloader, num_samples=100):
        """
        Comprehensive evaluation of the model
        """
        model.eval()
        
        all_generated = []
        all_real = []
        all_text_prompts = []
        clip_similarities = []
        
        with torch.no_grad():
            for i, (real_sketches, text_prompts, labels) in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break
                    
                real_sketches = real_sketches.to(self.device)
                
                # Generate sketches
                if hasattr(model, 'reparameterize'):  # VAE model
                    generated_sketches, mu, logvar, text_embeddings = model(text_prompts)
                else:  # Diffusion model (simplified)
                    # For diffusion, you'd implement proper sampling here
                    generated_sketches = torch.rand_like(real_sketches)
                
                # Collect samples
                all_generated.append(generated_sketches.cpu())
                all_real.append(real_sketches.cpu())
                all_text_prompts.extend(text_prompts)
                
                # Calculate CLIP similarity for this batch
                similarity = self.clip_similarity_score(generated_sketches, text_prompts)
                clip_similarities.append(similarity)
        
        # Concatenate all samples
        all_generated = torch.cat(all_generated, dim=0).to(self.device)
        all_real = torch.cat(all_real, dim=0).to(self.device)
        
        # Calculate metrics
        avg_clip_similarity = np.mean(clip_similarities)
        fid_score = self.calculate_fid(all_real, all_generated)
        
        results = {
            'clip_similarity': avg_clip_similarity,
            'fid_score': fid_score,
            'num_samples': len(all_generated)
        }
        
        return results


class TextToSketchTrainer:
    """Training pipeline for text-to-sketch models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.clip_loss = CLIPGuidedLoss().to(device)
        self.evaluator = EvaluationMetrics(device)
        
    def vae_loss(self, generated_sketches, target_sketches, mu, logvar, text_prompts, 
                 kl_weight=0.001, clip_weight=0.1):
        """Combined loss for VAE training"""
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(generated_sketches, target_sketches, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # CLIP guided loss
        clip_loss = self.clip_loss(generated_sketches, text_prompts)
        
        total_loss = recon_loss + kl_weight * kl_loss + clip_weight * clip_loss
        
        return total_loss, recon_loss, kl_loss, clip_loss
    
    def diffusion_loss(self, model_output, target_noise):
        """Simple MSE loss for diffusion training"""
        return F.mse_loss(model_output, target_noise)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (target_sketches, text_prompts, labels) in enumerate(dataloader):
            target_sketches = target_sketches.to(self.device)
            
            optimizer.zero_grad()
            
            if hasattr(self.model, 'reparameterize'):  # VAE model
                generated_sketches, mu, logvar, text_embeddings = self.model(text_prompts)
                loss, recon_loss, kl_loss, clip_loss = self.vae_loss(
                    generated_sketches, target_sketches, mu, logvar, text_prompts
                )
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Total Loss: {loss.item():.4f}, '
                          f'Recon: {recon_loss.item():.4f}, '
                          f'KL: {kl_loss.item():.4f}, '
                          f'CLIP: {clip_loss.item():.4f}')
            else:
                # Diffusion training would go here
                # This is a simplified placeholder
                loss = torch.tensor(0.0)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, num_samples=100):
        """Evaluate the model"""
        return self.evaluator.evaluate_model(self.model, dataloader, num_samples)


def create_model(model_type='vae'):
    """Factory function to create models"""
    if model_type == 'vae':
        return VAEBaseline()
    elif model_type == 'diffusion':
        return DiffusionModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
