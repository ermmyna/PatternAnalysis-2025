"""
Training script for Siamese Network on ISIC 2020
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from modules import EmbeddingNet, SiameseHead, ContrastiveLoss
from dataset import make_dataloader
from utils import (set_seed, compute_metrics, save_learning_curves, save_roc_curve,
                  save_confusion_matrix, plot_embeddings_umap, save_metrics_json,
                  create_output_dirs)


class LinearProbe(nn.Module):
    """Simple linear classifier on top of embeddings for evaluation."""
    
    def __init__(self, embed_dim, num_classes=2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class SingleImageDataset(Dataset):
    """Dataset wrapper to get single images with labels (not pairs)."""
    
    def __init__(self, pair_dataset):
        self.df = pair_dataset.df
        self.transform = pair_dataset.transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['full_path']
        label = row['label']
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, label


def train_epoch(embedding_net, siamese_head, contrastive_loss, train_loader, 
                optimizer_embed, device, epoch):
    """Train embedding network with contrastive loss for one epoch."""
    embedding_net.train()
    
    total_loss = 0
    num_batches = 0
    all_pos_dists = []
    all_neg_dists = []
    
    for batch_idx, (img1, img2, same) in enumerate(train_loader):
        img1, img2, same = img1.to(device), img2.to(device), same.to(device)
        
        # Print first batch info for debugging
        if epoch == 1 and batch_idx == 0:
            print(f"\n{'='*70}")
            print("FIRST BATCH DEBUG INFO:")
            print(f"{'='*70}")
            print(f"img1 shape: {img1.shape}")
            print(f"img2 shape: {img2.shape}")
            print(f"same shape: {same.shape}, dtype: {same.dtype}")
            print(f"\nPair labels (1=same class, 0=different class):")
            print(f"  First 8 pairs: {same[:min(8, len(same))].cpu().numpy()}")
            print(f"  Positive pairs: {same.sum().item()}/{len(same)}")
            print(f"  Negative pairs: {(len(same) - same.sum()).item()}/{len(same)}")
            print(f"{'='*70}\n")
        
        # Forward pass: get embeddings
        z1 = embedding_net(img1)
        z2 = embedding_net(img2)
        
        # Compute contrastive loss
        _, distances = siamese_head(z1, z2)
        loss, stats = contrastive_loss(distances, same)
        
        # Backward pass
        optimizer_embed.zero_grad()
        loss.backward()
        optimizer_embed.step()
        
        total_loss += loss.item()
        all_pos_dists.append(stats['pos_mean_d'])
        all_neg_dists.append(stats['neg_mean_d'])
        num_batches += 1
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}, "
                  f"PosD: {stats['pos_mean_d']:.3f}, NegD: {stats['neg_mean_d']:.3f}")
    
    avg_loss = total_loss / num_batches
    avg_pos_d = np.mean(all_pos_dists)
    avg_neg_d = np.mean(all_neg_dists)
    
    return avg_loss, avg_pos_d, avg_neg_d


def train_probe_epoch(embedding_net, probe, probe_criterion, train_dataset, 
                      optimizer_probe, device, batch_size=256):
    """Train linear probe on single images for one epoch."""
    embedding_net.eval()
    probe.train()
    
    # Create single image dataloader
    single_dataset = SingleImageDataset(train_dataset)
    single_loader = DataLoader(
        single_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    total_loss = 0
    num_batches = 0
    
    for imgs, labels in single_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Get embeddings (no grad for embedding net)
        with torch.no_grad():
            embeddings = embedding_net(imgs)
        
        # Forward through probe
        logits = probe(embeddings)
        loss = probe_criterion(logits, labels)
        
        # Backward pass
        optimizer_probe.zero_grad()
        loss.backward()
        optimizer_probe.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(embedding_net, probe, val_dataset, device, batch_size=128):
    """Evaluate linear probe on validation/test set."""
    embedding_net.eval()
    probe.eval()
    
    # Create single image dataloader
    single_dataset = SingleImageDataset(val_dataset)
    single_loader = DataLoader(
        single_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for imgs, labels in single_loader:
            imgs = imgs.to(device)
            
            # Get embeddings and predictions
            embeddings = embedding_net(imgs)
            logits = probe(embeddings)
            
            probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of class 1
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_embeddings = np.vstack(all_embeddings)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_embeddings, all_labels, all_probs, all_preds


def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network on ISIC 2020')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/val.csv')
    parser.add_argument('--test_csv', type=str, default='data/test.csv')
    parser.add_argument('--images_root', type=str, default='ISIC2020/train')
    parser.add_argument('--out_dir', type=str, 
                       default='recognition/siamese_isic2020_49324255/runs/exp1')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--unfreeze_after', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    create_output_dirs(args.out_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Using device: {device}")
    print(f"{'='*70}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = make_dataloader(
        csv_path=args.train_csv,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=True,
        images_root=args.images_root,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = make_dataloader(
        csv_path=args.val_csv,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        images_root=args.images_root,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    test_loader = make_dataloader(
        csv_path=args.test_csv,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        images_root=args.images_root,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Build models
    print(f"\nBuilding models...")
    print(f"  Backbone: {args.backbone}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Margin: {args.margin}")
    
    embedding_net = EmbeddingNet(
        backbone=args.backbone,
        out_dim=args.embed_dim,
        freeze_backbone=True
    ).to(device)
    
    siamese_head = SiameseHead(metric='cosine').to(device)
    contrastive_loss = ContrastiveLoss(margin=args.margin)
    
    # Linear probe for classification
    probe = LinearProbe(embed_dim=args.embed_dim, num_classes=2).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    
    # Optimizers
    optimizer_embed = optim.AdamW(
        embedding_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    optimizer_probe = optim.AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer_embed, T_max=args.epochs)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_val_auc = 0.0
    patience = 4
    patience_counter = 0
    
    train_losses = []
    val_metrics_history = []
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze backbone after warm-up
        if args.unfreeze_after > 0 and epoch == args.unfreeze_after + 1:
            print(f"\n{'='*70}")
            print(f">>> Unfreezing last block of backbone at epoch {epoch}")
            print(f"{'='*70}")
            embedding_net.unfreeze_last_block()
            # Lower learning rate for backbone
            optimizer_embed = optim.AdamW([
                {'params': embedding_net.projection.parameters(), 'lr': args.lr},
                {'params': embedding_net.backbone.parameters(), 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
        
        # Train embedding network with contrastive loss
        train_loss, pos_d, neg_d = train_epoch(
            embedding_net, siamese_head, contrastive_loss, 
            train_loader, optimizer_embed, device, epoch
        )
        
        # Train linear probe
        probe_loss = train_probe_epoch(
            embedding_net, probe, probe_criterion, 
            train_loader.dataset, optimizer_probe, device
        )
        
        # Validate
        val_metrics, _, _, _, _ = evaluate(
            embedding_net, probe, val_loader.dataset, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs} - Time: {epoch_time:.1f}s")
        print(f"{'='*70}")
        print(f"Train:")
        print(f"  Contrastive Loss: {train_loss:.4f}")
        print(f"  Probe Loss:       {probe_loss:.4f}")
        print(f"  Pos Distance:     {pos_d:.3f}")
        print(f"  Neg Distance:     {neg_d:.3f}")
        print(f"Validation:")
        print(f"  AUC:              {val_metrics['auc']:.4f}")
        print(f"  Accuracy:         {val_metrics['accuracy']:.4f}")
        print(f"  F1 Score:         {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'embedding_net': embedding_net.state_dict(),
                'probe': probe.state_dict(),
                'optimizer_embed': optimizer_embed.state_dict(),
                'optimizer_probe': optimizer_probe.state_dict(),
                'best_val_auc': best_val_auc,
                'args': vars(args)
            }
            
            checkpoint_path = os.path.join(args.out_dir, 'checkpoints', 'best.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ New best model saved! (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"\n✗ No improvement (patience: {patience_counter}/{patience})")
        
        print(f"{'='*70}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Save learning curves
    print("\nSaving learning curves...")
    curves_path = os.path.join(args.out_dir, 'figures', 'learning_curves.pdf')
    save_learning_curves(train_losses, val_metrics_history, curves_path)
    
    # Load best model for final evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoints', 'best.pt'))
    embedding_net.load_state_dict(checkpoint['embedding_net'])
    probe.load_state_dict(checkpoint['probe'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics, test_embeddings, test_labels, test_probs, test_preds = evaluate(
        embedding_net, probe, test_loader.dataset, device
    )
    
    print(f"\n{'='*70}")
    print("TEST RESULTS:")
    print(f"{'='*70}")
    print(f"AUC:      {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"{'='*70}")
    
    # Save test visualizations
    print("\nSaving test visualizations...")
    
    roc_path = os.path.join(args.out_dir, 'figures', 'roc_curve.pdf')
    save_roc_curve(test_labels, test_probs, roc_path)
    
    cm_path = os.path.join(args.out_dir, 'figures', 'confusion_matrix.pdf')
    save_confusion_matrix(test_labels, test_preds, cm_path)
    
    # Optional: UMAP embeddings
    umap_path = os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf')
    if len(test_embeddings) > 2000:
        idx = np.random.choice(len(test_embeddings), 2000, replace=False)
        plot_embeddings_umap(test_embeddings[idx], test_labels[idx], umap_path)
    else:
        plot_embeddings_umap(test_embeddings, test_labels, umap_path)
    
    # Save final metrics
    final_metrics = {
        'best_epoch': checkpoint['epoch'],
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_metrics['auc']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'total_training_time_minutes': float(total_time / 60),
        'hyperparameters': vars(args)
    }
    
    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    save_metrics_json(final_metrics, metrics_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print(f"Results saved to: {args.out_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()