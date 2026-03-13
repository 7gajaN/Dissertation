"""
FCS Predictor Network - Learns to predict Force Consistency Score from joint positions.
This allows physics-aware training via differentiable physics loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCSPredictor(nn.Module):
    """
    Predicts FCS (Force Consistency Score) from joint positions.
    
    Architecture:
    - Input: (B, S, 24, 3) joint positions
    - Flatten to (B, S, 72)
    - Temporal convolutions to extract motion patterns
    - Global pooling
    - MLP to predict scalar FCS
    
    This network is trained offline on real mocap data with ground-truth FCS,
    then used during training as a differentiable physics loss.
    """
    
    def __init__(self, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input: 24 joints * 3 coords = 72 features per frame
        input_dim = 24 * 3
        
        # Temporal feature extraction (1D convolutions over time)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of temporal conv blocks
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Velocity features (first derivative)
        self.vel_proj = nn.Linear(input_dim, hidden_dim // 2)
        
        # Acceleration features (second derivative)
        self.accel_proj = nn.Linear(input_dim, hidden_dim // 2)
        
        # Combine position, velocity, acceleration features
        combined_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        
        # MLP head to predict FCS
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # FCS is always positive
        )
        
    def forward(self, joint_positions):
        """
        Args:
            joint_positions: (B, S, 24, 3) tensor of joint positions
            
        Returns:
            fcs_pred: (B,) predicted FCS scores
        """
        B, S, J, C = joint_positions.shape
        
        # Flatten joints: (B, S, 72)
        x = joint_positions.reshape(B, S, -1)
        
        # Position features
        pos_feats = self.input_proj(x)  # (B, S, hidden_dim)
        pos_feats = pos_feats.transpose(1, 2)  # (B, hidden_dim, S)
        
        # Apply temporal convolutions
        for conv_block in self.conv_blocks:
            pos_feats = pos_feats + conv_block(pos_feats)  # Residual connection
        
        pos_feats = pos_feats.transpose(1, 2)  # (B, S, hidden_dim)
        
        # Velocity features (first derivative)
        vel = x[:, 1:] - x[:, :-1]  # (B, S-1, 72)
        vel_feats = self.vel_proj(vel)  # (B, S-1, hidden_dim//2)
        
        # Acceleration features (second derivative)
        accel = vel[:, 1:] - vel[:, :-1]  # (B, S-2, 72)
        accel_feats = self.accel_proj(accel)  # (B, S-2, hidden_dim//2)
        
        # Align sequence lengths (use S-2 to match acceleration)
        pos_feats = pos_feats[:, :S-2, :]
        vel_feats = vel_feats[:, :S-2, :]
        
        # Concatenate all features
        combined = torch.cat([pos_feats, vel_feats, accel_feats], dim=-1)  # (B, S-2, combined_dim)
        
        # Global average pooling over time
        pooled = combined.mean(dim=1)  # (B, combined_dim)
        
        # Predict FCS
        fcs_pred = self.head(pooled).squeeze(-1)  # (B,)
        
        return fcs_pred


class FCSPredictorLoss(nn.Module):
    """Loss for training FCS predictor"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_fcs, true_fcs):
        """
        Args:
            pred_fcs: (B,) predicted FCS scores
            true_fcs: (B,) ground-truth FCS scores
            
        Returns:
            loss: scalar loss
        """
        # MSE loss
        mse_loss = F.mse_loss(pred_fcs, true_fcs)
        
        # Relative error loss (helps with varying magnitudes)
        rel_error = torch.abs(pred_fcs - true_fcs) / (true_fcs + 1e-6)
        rel_loss = rel_error.mean()
        
        # Combined loss
        total_loss = mse_loss + 0.1 * rel_loss
        
        return total_loss
