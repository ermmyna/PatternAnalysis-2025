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

from modules import EmbeddingNet, SiameseHead, ContrastiveLoss
from dataset import make_dataloader
from utils import (set_seed, compute_metrics, save_learning_curves, save_roc_curve,
                  save_confusion_matrix, plot_embeddings_umap, save_metrics_json,
                  create_output_dirs)


class LinearProbe(nn.Module):
    """Simple linear classifier on top of frozen embeddings."""
    
    def __init__(self, embed_dim, num_classes=2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_epoch(embedding_net, probe, siamese_head, contrastive_loss, probe_criterion,
                train_loader, optimizer_embed, optimizer_probe, device, epoch):
    """Train for one epoch."""
    embedding_net.train()
    probe.train()
    
    total_loss = 0
    total_probe_loss = 0
    num_batches = 0
    all_pos_dists = []
    all_neg_dists = []
    
    for batch_idx, (img1, img2, target, label1, label2) in enumerate(train_loader):
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        label1, label2 = label1.to(device), label2.to(device)
        
        # Print first batch info for debugging
        if epoch == 1 and batch_idx == 0:
            print(f"\nFirst batch shapes:")
            print(f"  img1: {img1.shape}, img2: {img2.shape}")
            print(f"  target (pair labels): {target[:8]}")
            print(f"  label1 (class): {label1[:8]}")
            print(f"  label2 (class): {label2[:8]}")
        
        # Forward pass: get embeddings
        z1 = embedding_net(img1)
        z2 = embedding_net(img2)
        
        # Compute contrastive loss
        _, distances = siamese_head(z1, z2)
        cont_loss, stats = contrastive_loss(distances, target)
        
        # Update embedding network
        optimizer_embed.zero_grad()
        cont_loss.backward()
        optimizer_embed.step()
        
        total_loss += cont_loss.item()
        all_pos_dists.append(stats['pos_mean_d'])
        all_neg_dists.append(stats['neg_mean_d'])
        
        # Train linear probe on embeddings (use both sides)
        with torch.no_grad():
            z1_detached = embedding_net(img1)
            z2_detached = embedding_net(img2)
        
        # Concatenate embeddings and labels from both sides
        probe_embeddings = torch.cat([z1_detached, z2_detached], dim=0)
        probe_labels = torch.cat([label1, label2], dim=0)
        
        # Forward through probe
        probe_logits = probe(probe_embeddings)
        probe_loss = probe_criterion(probe_logits, probe_labels)
        
        # Update probe
        optimizer_probe.zero_grad()
        probe_loss.backward()
        optimizer_probe.step()
        
        total_probe_loss += probe_loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"ContLoss: {cont_loss.item():.4f}, ProbeLoss: {probe_loss.item():.4f}, "
                  f"PosD: {stats['pos_mean_d']:.3f}, NegD: {stats['neg_mean_d']:.3f}")
    
    avg_loss = total_loss / num_batches
    avg_probe_loss = total_probe_loss / num_batches
    avg_pos_d = np.mean(all_pos_dists)
    avg_neg_d = np.mean(all_neg_dists)
    
    return avg_loss, avg_probe_loss, avg_pos_d, avg_neg_d


def evaluate(embedding_net, probe, val_loader, device):
    """Evaluate linear probe on validation set."""
    embedding_net.eval()
    probe.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for img1, img2, target, label1, label2 in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            
            # Get embeddings and predictions for both images
            z1 = embedding_net(img1)
            z2 = embedding_net(img2)
            
            logits1 = probe(z1)
            logits2 = probe(z2)
            
            probs1 = torch.softmax(logits1, dim=1)[:, 1]  # Prob of malignant
            probs2 = torch.softmax(logits2, dim=1)[:, 1]
            
            preds1 = torch.argmax(logits1, dim=1)
            preds2 = torch.argmax(logits2, dim=1)
            
            # Collect predictions from both sides
            all_labels.extend(label1.cpu().numpy())
            all_labels.extend(label2.cpu().numpy())
            all_preds.extend(preds1.cpu().numpy())
            all_preds.extend(preds2.cpu().numpy())
            all_probs.extend(probs1.cpu().numpy())
            all_probs.extend(probs2.cpu().numpy())
            all_embeddings.append(z1.cpu().numpy())
            all_embeddings.append(z2.cpu().numpy())
    
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
    parser.add_argument('--train_csv', type=str, default='data/train.csv',
                       help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='data/val.csv',
                       help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='data/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--images_root', type=str, default='ISIC2020/train',
                       help='Root directory for images')
    parser.add_argument('--out_dir', type=str, 
                       default='recognition/siamese_isic2020_yourID/runs/exp1',
                       help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                       help='Backbone architecture')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Contrastive loss margin')
    parser.add_argument('--unfreeze_after', type=int, default=0,
                       help='Epochs before unfreezing backbone (0 = never)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    create_output_dirs(args.out_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    
    # Build models
    print(f"\nBuilding models with backbone: {args.backbone}")
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
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
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
            print(f"\n>>> Unfreezing last block of backbone at epoch {epoch}")
            embedding_net.unfreeze_last_block()
            # Lower learning rate for backbone
            optimizer_embed = optim.AdamW([
                {'params': embedding_net.projection.parameters(), 'lr': args.lr},
                {'params': embedding_net.backbone.parameters(), 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
        
        # Train
        train_loss, probe_loss, pos_d, neg_d = train_epoch(
            embedding_net, probe, siamese_head, contrastive_loss, probe_criterion,
            train_loader, optimizer_embed, optimizer_probe, device, epoch
        )
        
        # Validate
        val_metrics, _, _, _, _ = evaluate(embedding_net, probe, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - ContLoss: {train_loss:.4f}, ProbeLoss: {probe_loss:.4f}, "
              f"PosD: {pos_d:.3f}, NegD: {neg_d:.3f}")
        print(f"  Val   - AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
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
            print(f"  >>> New best model saved! (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  >>> No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
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
        embedding_net, probe, test_loader, device
    )
    
    print(f"\nTest Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Save test visualizations
    print("\nSaving test visualizations...")
    
    roc_path = os.path.join(args.out_dir, 'figures', 'roc_curve.pdf')
    save_roc_curve(test_labels, test_probs, roc_path)
    
    cm_path = os.path.join(args.out_dir, 'figures', 'confusion_matrix.pdf')
    save_confusion_matrix(test_labels, test_preds, cm_path)
    
    # Optional: UMAP embeddings
    umap_path = os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf')
    # Subsample for faster plotting
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
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: {args.out_dir}")
    print("="*60)


if __name__ == '__main__':
    main()