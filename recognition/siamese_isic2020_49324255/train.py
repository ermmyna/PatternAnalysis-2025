"""
Training script for Siamese Network on ISIC 2020
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from modules import EmbeddingNet, SiameseHead, ContrastiveLoss
from dataset import make_dataloader
from utils import (set_seed, compute_metrics, save_learning_curves, save_roc_curve,
                  save_confusion_matrix, plot_embeddings_umap, save_metrics_json,
                  create_output_dirs)


class LinearProbe(nn.Module):
    """Simple linear classifier: embed_dim -> 2 classes."""
    
    def __init__(self, embed_dim, num_classes=2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class SingleImageDataset(Dataset):
    """Wraps pair dataset to return single images with labels."""
    
    def __init__(self, pair_dataset):
        self.df = pair_dataset.df
        self.transform = pair_dataset.transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['full_path']).convert('RGB')
        return self.transform(img), row['label']


def choose_threshold(y_true, y_prob, metric='f1'):
    """
    Find optimal threshold by sweeping 0.01 to 0.99.
    Returns: (best_threshold, {'best_f1', 'precision_at_best', 'recall_at_best'})
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_score = 0.0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        score = f1 if metric == 'f1' else (prec if metric == 'precision' else rec)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'best_f1': f1,
                'precision_at_best': prec,
                'recall_at_best': rec
            }
    
    return best_threshold, best_metrics


def save_pr_curve(y_true, y_prob, save_path):
    """Save Precision-Recall curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PR curve saved to {save_path}")
    except Exception as e:
        print(f"Could not save PR curve: {e}")


def train_epoch(embedding_net, siamese_head, contrastive_loss, train_loader, optimizer, device, epoch):
    """Train embedding network with contrastive loss."""
    embedding_net.train()
    total_loss = 0
    all_pos_d = []
    all_neg_d = []
    
    for batch_idx, (img1, img2, same) in enumerate(train_loader):
        img1, img2, same = img1.to(device), img2.to(device), same.to(device)
        
        # Debug first batch
        if epoch == 1 and batch_idx == 0:
            print(f"\n{'='*70}")
            print("FIRST BATCH:")
            print(f"  img1: {img1.shape}, img2: {img2.shape}, same: {same.shape}")
            print(f"  Pair labels (1=same, 0=diff): {same[:8].cpu().numpy()}")
            print(f"  Pos/Neg: {same.sum().item()}/{(len(same)-same.sum()).item()}")
            print(f"{'='*70}\n")
        
        # Forward
        z1, z2 = embedding_net(img1), embedding_net(img2)
        _, distances = siamese_head(z1, z2)
        loss, stats = contrastive_loss(distances, same)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_pos_d.append(stats['pos_mean_d'])
        all_neg_d.append(stats['neg_mean_d'])
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}, PosD: {stats['pos_mean_d']:.3f}, NegD: {stats['neg_mean_d']:.3f}")
    
    return total_loss / len(train_loader), np.mean(all_pos_d), np.mean(all_neg_d)


def train_probe(embedding_net, probe, criterion, dataset, optimizer, device):
    """Train linear probe on single images."""
    embedding_net.eval()
    probe.train()
    
    loader = DataLoader(SingleImageDataset(dataset), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    total_loss = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            emb = embedding_net(imgs)
        
        logits = probe(emb)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(embedding_net, probe, dataset, device, threshold=0.5):
    """Evaluate probe, return metrics and arrays."""
    embedding_net.eval()
    probe.eval()
    
    loader = DataLoader(SingleImageDataset(dataset), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    all_labels, all_probs, all_embs = [], [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = embedding_net(imgs)
            probs = torch.softmax(probe(emb), dim=1)[:, 1]
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embs.append(emb.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)
    all_embs = np.vstack(all_embs)
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['aucpr'] = average_precision_score(all_labels, all_probs)
    
    return metrics, all_embs, all_labels, all_probs, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='data/train.csv')
    parser.add_argument('--val_csv', default='data/val.csv')
    parser.add_argument('--test_csv', default='data/test.csv')
    parser.add_argument('--images_root', default='ISIC2020/train')
    parser.add_argument('--out_dir', default='recognition/siamese_isic2020_49324255/runs/exp1')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--backbone', default='efficientnet_b0')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--unfreeze_after', type=int, default=0)
    parser.add_argument('--pos_ratio', type=float, default=0.5,
                       help='Probability of positive pairs (default 0.5 for balanced)')
    parser.add_argument('--auto_threshold', action='store_true', default=False,
                       help='Auto-compute optimal threshold on validation when epochs=0')
    args = parser.parse_args()
    
    set_seed(args.seed)
    create_output_dirs(args.out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Device: {device} | Backbone: {args.backbone} | Embed dim: {args.embed_dim}")
    print(f"Pos ratio: {args.pos_ratio}")
    print(f"{'='*70}")
    
    # Data loaders with pos_ratio
    train_loader = make_dataloader(args.train_csv, args.batch_size, args.img_size, True, 
                                   args.images_root, args.num_workers, pos_ratio=args.pos_ratio)
    val_loader = make_dataloader(args.val_csv, args.batch_size, args.img_size, False, 
                                 args.images_root, args.num_workers, pos_ratio=args.pos_ratio)
    test_loader = make_dataloader(args.test_csv, args.batch_size, args.img_size, False, 
                                  args.images_root, args.num_workers, pos_ratio=args.pos_ratio)
    
    print(f"\nTrain: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Models
    embedding_net = EmbeddingNet(args.backbone, args.embed_dim, freeze_backbone=True).to(device)
    siamese_head = SiameseHead(metric='cosine').to(device)
    contrastive_loss = ContrastiveLoss(margin=args.margin)
    probe = LinearProbe(args.embed_dim, 2).to(device)
    probe_criterion = nn.CrossEntropyLoss()
    
    # Optimizers
    opt_embed = optim.AdamW(embedding_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_probe = optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt_embed, T_max=args.epochs)
    
    # Check for eval-only mode
    ckpt_path = os.path.join(args.out_dir, 'checkpoints', 'best.pt')
    metrics_json_path = os.path.join(args.out_dir, 'metrics.json')
    
    if args.epochs == 0 and args.auto_threshold:
        print(f"\n{'='*70}")
        print("EVAL-ONLY MODE: Auto-computing optimal threshold")
        print(f"{'='*70}")
        
        if not os.path.exists(ckpt_path):
            print(f"ERROR: No checkpoint found at {ckpt_path}")
            return
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Handle both key names
        if 'embedding_net' in ckpt:
            embedding_net.load_state_dict(ckpt['embedding_net'])
            probe.load_state_dict(ckpt['probe'])
        elif 'model' in ckpt:
            embedding_net.load_state_dict(ckpt['model'])
            probe.load_state_dict(ckpt.get('probe', probe.state_dict()))
        else:
            print("ERROR: Checkpoint must contain 'embedding_net' or 'model' key")
            return
        
        # Get or compute threshold
        best_thresh = ckpt.get('best_threshold', None)
        if best_thresh is None and os.path.exists(metrics_json_path):
            with open(metrics_json_path, 'r') as f:
                best_thresh = json.load(f).get('best_threshold', None)
        
        if best_thresh is None:
            print("\nComputing optimal threshold on validation...")
            val_metrics, _, val_labels, val_probs, _ = evaluate(embedding_net, probe, val_loader.dataset, device, 0.5)
            best_thresh, thresh_metrics = choose_threshold(val_labels, val_probs, 'f1')
            
            print(f"\nOptimal Threshold: {best_thresh:.3f}")
            print(f"  F1:        {thresh_metrics['best_f1']:.4f}")
            print(f"  Precision: {thresh_metrics['precision_at_best']:.4f}")
            print(f"  Recall:    {thresh_metrics['recall_at_best']:.4f}")
            
            save_pr_curve(val_labels, val_probs, os.path.join(args.out_dir, 'figures', 'pr_curve_val.pdf'))
            
            metrics_to_save = {
                'best_threshold': float(best_thresh),
                'val_best_f1': float(thresh_metrics['best_f1']),
                'val_precision_at_best': float(thresh_metrics['precision_at_best']),
                'val_recall_at_best': float(thresh_metrics['recall_at_best']),
                'val_auc': float(val_metrics['auc']),
                'val_aucpr': float(val_metrics['aucpr'])
            }
        else:
            print(f"\nUsing existing threshold: {best_thresh:.3f}")
            metrics_to_save = {'best_threshold': float(best_thresh)}
        
        # Test evaluation
        test_metrics, test_embs, test_labels, test_probs, test_preds = evaluate(
            embedding_net, probe, test_loader.dataset, device, best_thresh
        )
        
        print(f"\n{'='*70}")
        print("TEST RESULTS:")
        print(f"AUC={test_metrics['auc']:.4f}, AUCPR={test_metrics['aucpr']:.4f}, "
              f"Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")
        print(f"{'='*70}")
        
        save_pr_curve(test_labels, test_probs, os.path.join(args.out_dir, 'figures', 'pr_curve_test.pdf'))
        save_roc_curve(test_labels, test_probs, os.path.join(args.out_dir, 'figures', 'roc_curve.pdf'))
        save_confusion_matrix(test_labels, test_preds, os.path.join(args.out_dir, 'figures', 'confusion_matrix.pdf'))
        
        if len(test_embs) > 2000:
            idx = np.random.choice(len(test_embs), 2000, replace=False)
            plot_embeddings_umap(test_embs[idx], test_labels[idx], os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf'))
        else:
            plot_embeddings_umap(test_embs, test_labels, os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf'))
        
        metrics_to_save.update({
            'test_auc': float(test_metrics['auc']),
            'test_aucpr': float(test_metrics['aucpr']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1_at_best_thr': float(test_metrics['f1']),
            'hyperparameters': vars(args)
        })
        
        save_metrics_json(metrics_to_save, metrics_json_path)
        print(f"\n✓ Eval-only complete!\n")
        return
    
    # Training
    print(f"\n{'='*70}\nSTARTING TRAINING\n{'='*70}")
    best_auc = 0.0
    best_thresh = 0.5
    patience_counter = 0
    train_losses = []
    val_metrics_hist = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Unfreeze
        if args.unfreeze_after > 0 and epoch == args.unfreeze_after + 1:
            print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            embedding_net.unfreeze_last_block()
            opt_embed = optim.AdamW([
                {'params': embedding_net.projection.parameters(), 'lr': args.lr},
                {'params': embedding_net.backbone.parameters(), 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
        
        # Train
        loss, pos_d, neg_d = train_epoch(embedding_net, siamese_head, contrastive_loss, train_loader, opt_embed, device, epoch)
        probe_loss = train_probe(embedding_net, probe, probe_criterion, train_loader.dataset, opt_probe, device)
        
        # Validate
        val_metrics, _, val_labels, val_probs, _ = evaluate(embedding_net, probe, val_loader.dataset, device)
        thresh, thresh_metrics = choose_threshold(val_labels, val_probs, 'f1')
        
        scheduler.step()
        train_losses.append(loss)
        val_metrics_hist.append(val_metrics)
        
        # Print
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"Train: Loss={loss:.4f}, ProbeLoss={probe_loss:.4f}, PosD={pos_d:.3f}, NegD={neg_d:.3f}")
        print(f"Val (0.5): AUC={val_metrics['auc']:.4f}, AUCPR={val_metrics['aucpr']:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
        print(f"Optimal thresh={thresh:.3f}: F1={thresh_metrics['best_f1']:.4f}, Prec={thresh_metrics['precision_at_best']:.4f}, Rec={thresh_metrics['recall_at_best']:.4f}")
        
        # Save best
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_thresh = thresh
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'embedding_net': embedding_net.state_dict(),
                'probe': probe.state_dict(),
                'best_val_auc': best_auc,
                'best_threshold': best_thresh,
                'args': vars(args)
            }, ckpt_path)
            print(f"✓ Best model saved! AUC={best_auc:.4f}, Thresh={best_thresh:.3f}")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/4)")
        
        print(f"{'='*70}")
        
        if patience_counter >= 4:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min")
    
    # Save curves
    save_learning_curves(train_losses, val_metrics_hist, os.path.join(args.out_dir, 'figures', 'learning_curves.pdf'))
    
    # Load best and test
    if not os.path.exists(ckpt_path):
        print(f"\nNo checkpoint found. Skipping test evaluation.")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    embedding_net.load_state_dict(ckpt['embedding_net'])
    probe.load_state_dict(ckpt['probe'])
    best_thresh = ckpt.get('best_threshold', 0.5)
    
    print(f"\nTesting with threshold={best_thresh:.3f}...")
    test_metrics, test_embs, test_labels, test_probs, test_preds = evaluate(embedding_net, probe, test_loader.dataset, device, best_thresh)
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS:")
    print(f"AUC={test_metrics['auc']:.4f}, AUCPR={test_metrics['aucpr']:.4f}, Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")
    print(f"{'='*70}")
    
    # Save visualizations
    save_roc_curve(test_labels, test_probs, os.path.join(args.out_dir, 'figures', 'roc_curve.pdf'))
    save_pr_curve(test_labels, test_probs, os.path.join(args.out_dir, 'figures', 'pr_curve_test.pdf'))
    save_confusion_matrix(test_labels, test_preds, os.path.join(args.out_dir, 'figures', 'confusion_matrix.pdf'))
    
    if len(test_embs) > 2000:
        idx = np.random.choice(len(test_embs), 2000, replace=False)
        plot_embeddings_umap(test_embs[idx], test_labels[idx], os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf'))
    else:
        plot_embeddings_umap(test_embs, test_labels, os.path.join(args.out_dir, 'figures', 'embeddings_umap.pdf'))
    
    # Save metrics
    save_metrics_json({
        'best_epoch': ckpt['epoch'],
        'best_val_auc': float(best_auc),
        'best_threshold': float(best_thresh),
        'test_auc': float(test_metrics['auc']),
        'test_aucpr': float(test_metrics['aucpr']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_at_best_thr': float(test_metrics['f1']),
        'training_time_minutes': float((time.time() - start_time) / 60),
        'hyperparameters': vars(args)
    }, metrics_json_path)
    
    print(f"\n✓ Done! Results in {args.out_dir}\n")


if __name__ == '__main__':
    main()