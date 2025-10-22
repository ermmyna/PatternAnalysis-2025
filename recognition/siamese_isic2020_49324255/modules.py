"""
Siamese Network Modules for ISIC 2020
Embedding network, Siamese head, and Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_backbone(name="efficientnet_b0"):
    """
    Load a pretrained backbone from timm and return (backbone, feature_dim).
    
    Args:
        name: backbone architecture name (e.g., 'efficientnet_b0', 'resnet50')
    
    Returns:
        backbone: nn.Module without classifier head
        feature_dim: int, dimension of backbone output features
    """
    # Load pretrained model
    model = timm.create_model(name, pretrained=True, num_classes=0)
    
    # Get feature dimension
    if hasattr(model, 'num_features'):
        feature_dim = model.num_features
    elif hasattr(model, 'feature_info'):
        feature_dim = model.feature_info[-1]['num_chs']
    else:
        # Fallback: pass dummy input
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = model(dummy)
            feature_dim = out.shape[1]
    
    return model, feature_dim


class EmbeddingNet(nn.Module):
    """
    Embedding network with pretrained backbone and projection head.
    Outputs L2-normalized embeddings.
    """
    
    def __init__(self, backbone="efficientnet_b0", out_dim=128, freeze_backbone=True):
        """
        Args:
            backbone: name of timm backbone architecture
            out_dim: dimension of output embeddings
            freeze_backbone: if True, freeze backbone weights initially
        """
        super(EmbeddingNet, self).__init__()
        
        self.backbone_name = backbone
        self.out_dim = out_dim
        
        # Build backbone
        self.backbone, feature_dim = build_backbone(backbone)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head: BN -> ReLU -> Linear
        self.projection = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, out_dim)
        )
        
    def forward(self, x):
        """
        Forward pass with L2 normalization.
        
        Args:
            x: input images (B, 3, H, W)
        
        Returns:
            z: L2-normalized embeddings (B, out_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to embedding space
        z = self.projection(features)
        
        # L2 normalize
        z = F.normalize(z, p=2, dim=1)
        
        return z
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_last_block(self):
        """Unfreeze only the last block of backbone for gradual unfreezing."""
        # This is backbone-specific; here's a generic approach
        all_params = list(self.backbone.parameters())
        # Unfreeze last 25% of parameters
        num_to_unfreeze = max(1, len(all_params) // 4)
        for param in all_params[-num_to_unfreeze:]:
            param.requires_grad = True


class SiameseHead(nn.Module):
    """
    Siamese head that computes similarity and distance between embedding pairs.
    """
    
    def __init__(self, metric="cosine"):
        """
        Args:
            metric: 'cosine' or 'euclidean'
        """
        super(SiameseHead, self).__init__()
        self.metric = metric
    
    def forward(self, z1, z2):
        """
        Compute similarity and distance between embedding pairs.
        
        Args:
            z1: embeddings of first images (B, D)
            z2: embeddings of second images (B, D)
        
        Returns:
            similarity: similarity scores (B,)
            distance: distance values (B,) - used for contrastive loss
        """
        if self.metric == "cosine":
            # Cosine similarity (already L2 normalized in EmbeddingNet)
            similarity = (z1 * z2).sum(dim=1)
            # Distance for contrastive loss: 1 - cosine_sim
            distance = 1.0 - similarity
        elif self.metric == "euclidean":
            # Euclidean distance
            distance = torch.sqrt(((z1 - z2) ** 2).sum(dim=1) + 1e-8)
            # Similarity (inverse of distance)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return similarity, distance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    L = y * d^2 + (1 - y) * max(0, margin - d)^2
    where y=1 for same class (positive pairs), y=0 for different class (negative pairs)
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, target):
        """
        Compute contrastive loss.
        
        Args:
            distance: distances between embedding pairs (B,)
            target: binary labels (B,) - 1 for same class, 0 for different class
        
        Returns:
            loss: scalar loss value
            stats: dict with 'pos_mean_d' and 'neg_mean_d'
        """
        # Positive pairs (same class): minimize distance
        pos_loss = target * (distance ** 2)
        
        # Negative pairs (different class): maximize distance up to margin
        neg_loss = (1 - target) * torch.clamp(self.margin - distance, min=0) ** 2
        
        # Total loss
        loss = (pos_loss + neg_loss).mean()
        
        # Compute statistics
        pos_mask = target == 1
        neg_mask = target == 0
        
        stats = {}
        if pos_mask.any():
            stats['pos_mean_d'] = distance[pos_mask].mean().item()
        else:
            stats['pos_mean_d'] = 0.0
        
        if neg_mask.any():
            stats['neg_mean_d'] = distance[neg_mask].mean().item()
        else:
            stats['neg_mean_d'] = 0.0
        
        return loss, stats