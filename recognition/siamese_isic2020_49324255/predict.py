"""
Prediction script for Siamese Network on ISIC 2020
Loads trained model and runs inference on test set
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from modules import EmbeddingNet
from train import LinearProbe


class ISICInferenceDataset(Dataset):
    """Simple dataset for inference (no pairs)."""
    
    def __init__(self, csv_path, img_size=224, images_root=None):
        self.img_size = img_size
        self.images_root = images_root
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Resolve paths
        if images_root is not None:
            from pathlib import Path
            self.df['full_path'] = self.df['image_path'].apply(
                lambda p: str(Path(images_root) / Path(p).name)
            )
        else:
            self.df['full_path'] = self.df['image_path']
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['full_path']
        label = row['label'] if 'label' in row else -1
        image_name = row['image_name']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, label, image_name


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained Siamese model')
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--images_root', type=str, required=True,
                       help='Root directory for images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']
    
    # Build models
    print(f"Building models...")
    embedding_net = EmbeddingNet(
        backbone=model_args['backbone'],
        out_dim=model_args['embed_dim'],
        freeze_backbone=False
    ).to(device)
    
    probe = LinearProbe(embed_dim=model_args['embed_dim'], num_classes=2).to(device)
    
    # Load weights
    embedding_net.load_state_dict(checkpoint['embedding_net'])
    probe.load_state_dict(checkpoint['probe'])
    
    embedding_net.eval()
    probe.eval()
    
    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    print(f"Best validation AUC: {checkpoint['best_val_auc']:.4f}")
    
    # Create dataset
    print(f"\nLoading data from {args.csv}")
    dataset = ISICInferenceDataset(
        csv_path=args.csv,
        img_size=args.img_size,
        images_root=args.images_root
    )
    
    # Limit to max_samples
    if len(dataset) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples (out of {len(dataset)})")
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset.df = dataset.df.iloc[indices].reset_index(drop=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run inference
    print(f"\nRunning inference on {len(dataset)} samples...")
    
    all_image_names = []
    all_labels = []
    all_probs = []
    all_preds = []
    all_embeddings = []
    
    with torch.no_grad():
        for imgs, labels, image_names in dataloader:
            imgs = imgs.to(device)
            
            # Get embeddings
            embeddings = embedding_net(imgs)
            
            # Get predictions
            logits = probe(embeddings)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of melanoma (class 1)
            preds = torch.argmax(logits, dim=1)
            
            all_image_names.extend(image_names)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'image_name': all_image_names,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'prob_melanoma': all_probs
    })
    
    # Save predictions
    os.makedirs(args.out_dir, exist_ok=True)
    predictions_path = os.path.join(args.out_dir, 'predictions.csv')
    results_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")
    
    # Print sample predictions
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    # Show a few positive and negative examples
    print("\nFirst 10 samples:")
    print(results_df.head(10).to_string(index=False))
    
    # If labels available, show some statistics
    if (results_df['true_label'] >= 0).all():
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        acc = accuracy_score(results_df['true_label'], results_df['predicted_label'])
        auc = roc_auc_score(results_df['true_label'], results_df['prob_melanoma'])
        f1 = f1_score(results_df['true_label'], results_df['predicted_label'])
        
        print(f"\n" + "="*80)
        print(f"Performance Metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print("="*80)
    
    # Show some high and low confidence predictions
    print(f"\n" + "="*80)
    print("High confidence melanoma predictions (top 5):")
    print("="*80)
    top_melanoma = results_df.nlargest(5, 'prob_melanoma')
    print(top_melanoma[['image_name', 'true_label', 'prob_melanoma']].to_string(index=False))
    
    print(f"\n" + "="*80)
    print("High confidence benign predictions (bottom 5):")
    print("="*80)
    top_benign = results_df.nsmallest(5, 'prob_melanoma')
    print(top_benign[['image_name', 'true_label', 'prob_melanoma']].to_string(index=False))
    
    print(f"\n" + "="*80)
    print("Inference complete!")
    print("="*80)


if __name__ == '__main__':
    main()