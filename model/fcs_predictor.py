"""
FCS Predictor Network - Learns to predict Force Consistency Score from joint positions.

The architecture explicitly computes physics-relevant features (CoM, acceleration,
foot contacts, force balance) and learns to combine them, rather than trying to
learn physics from scratch via generic convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# SMPL segment mass fractions (Winter 2009) — same as eval_fcs.py
SEGMENT_JOINT_MAPPING = {
    'head': ([15], 0.081),
    'trunk': ([0, 3, 6, 9, 12], 0.497),
    'upper_arm_r': ([17, 19], 0.028),
    'upper_arm_l': ([16, 18], 0.028),
    'forearm_r': ([19, 21], 0.016),
    'forearm_l': ([18, 20], 0.016),
    'hand_r': ([23], 0.006),
    'hand_l': ([22], 0.006),
    'thigh_r': ([2, 5], 0.100),
    'thigh_l': ([1, 4], 0.100),
    'shank_r': ([5, 8], 0.0465),
    'shank_l': ([4, 7], 0.0465),
    'foot_r': ([8, 11], 0.0145),
    'foot_l': ([7, 10], 0.0145),
}

FOOT_INDICES = [7, 8, 10, 11]  # L_Ankle, R_Ankle, L_Toe, R_Toe


class PhysicsFeatureExtractor(nn.Module):
    """
    Differentiable extraction of physics features that mirror FCS computation.
    All operations are differentiable so gradients flow back to the motion.
    """

    def __init__(self, fps=30, gravity=9.81, body_mass=70.0):
        super().__init__()
        self.dt = 1.0 / fps
        self.gravity = gravity
        self.body_mass = body_mass

        # Pre-compute segment mass weights as a buffer
        joint_weights = torch.zeros(24)
        joint_counts = torch.zeros(24)
        for segment_name, (joint_indices, mass_fraction) in SEGMENT_JOINT_MAPPING.items():
            for ji in joint_indices:
                joint_weights[ji] += mass_fraction / len(joint_indices)
                joint_counts[ji] += 1
        # Normalize so weights sum to 1
        joint_weights = joint_weights / joint_weights.sum()
        self.register_buffer('joint_mass_weights', joint_weights)

        # Learnable contact thresholds (initialized to reasonable values)
        self.height_threshold = nn.Parameter(torch.tensor(0.08))
        self.velocity_threshold = nn.Parameter(torch.tensor(0.20))

    def forward(self, joint_positions):
        """
        Extract physics features from joint positions.

        Args:
            joint_positions: (B, S, 24, 3)

        Returns:
            features: (B, S-2, F) physics features per frame
        """
        B, S, J, C = joint_positions.shape

        # ── 1. Center of Mass ──
        # Weighted average of joint positions
        weights = self.joint_mass_weights.view(1, 1, J, 1)  # (1, 1, 24, 1)
        com = (joint_positions * weights).sum(dim=2)  # (B, S, 3)

        # ── 2. CoM velocity and acceleration ──
        com_vel = (com[:, 1:] - com[:, :-1]) / self.dt  # (B, S-1, 3)
        com_acc = (com_vel[:, 1:] - com_vel[:, :-1]) / self.dt  # (B, S-2, 3)

        # ── 3. Required force (F = ma + weight) ──
        required_force = self.body_mass * com_acc  # (B, S-2, 3)
        required_force[:, :, 2] = required_force[:, :, 2] + self.body_mass * self.gravity

        required_force_mag = torch.norm(required_force, dim=-1, keepdim=True)  # (B, S-2, 1)

        # ── 4. Foot positions and contacts ──
        feet = joint_positions[:, :, FOOT_INDICES, :]  # (B, S, 4, 3)

        # Ground level: minimum foot height per sequence
        min_height = feet[:, :, :, 2].min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # (B, 1, 1)
        foot_height = feet[:, 1:-1, :, 2] - min_height.squeeze(-1).unsqueeze(1)  # (B, S-2, 4)

        # Foot velocity (horizontal)
        foot_vel = torch.norm(
            feet[:, 2:, :, :2] - feet[:, 1:-1, :, :2], dim=-1
        ) / self.dt  # (B, S-2, 4)

        # Soft contact detection (differentiable sigmoid instead of hard threshold)
        near_ground = torch.sigmoid(10.0 * (self.height_threshold - foot_height))  # (B, S-2, 4)
        slow_moving = torch.sigmoid(10.0 * (self.velocity_threshold - foot_vel))  # (B, S-2, 4)
        contact_prob = near_ground * slow_moving  # (B, S-2, 4)
        num_contacts = contact_prob.sum(dim=-1, keepdim=True)  # (B, S-2, 1)

        # ── 5. Available force (proportional to contacts) ──
        max_force_per_foot = self.body_mass * self.gravity * 3.0
        available_force = num_contacts * max_force_per_foot  # (B, S-2, 1)

        # ── 6. Force deficit (the core of FCS) ──
        force_deficit = F.relu(required_force_mag - available_force)  # (B, S-2, 1)
        normalized_deficit = force_deficit / (self.body_mass * self.gravity)

        # ── 7. Additional physics features ──
        # Joint velocities (all joints)
        joint_vel = torch.norm(
            joint_positions[:, 1:] - joint_positions[:, :-1], dim=-1
        ) / self.dt  # (B, S-1, 24)
        joint_vel = joint_vel[:, :-1]  # align to S-2

        # Joint accelerations
        joint_acc_raw = joint_positions[:, 2:] - 2 * joint_positions[:, 1:-1] + joint_positions[:, :-2]
        joint_acc = torch.norm(joint_acc_raw, dim=-1) / (self.dt ** 2)  # (B, S-2, 24)

        # Mean/max joint velocity and acceleration
        mean_joint_vel = joint_vel.mean(dim=-1, keepdim=True)  # (B, S-2, 1)
        max_joint_vel = joint_vel.max(dim=-1, keepdim=True)[0]
        mean_joint_acc = joint_acc.mean(dim=-1, keepdim=True)
        max_joint_acc = joint_acc.max(dim=-1, keepdim=True)[0]

        # Foot skating: velocity when in contact
        foot_skating = (foot_vel * contact_prob).mean(dim=-1, keepdim=True)  # (B, S-2, 1)

        # Ground penetration (soft)
        penetration = F.relu(-foot_height).mean(dim=-1, keepdim=True)  # (B, S-2, 1)

        # CoM height
        com_height = com[:, 1:-1, 2:3]  # (B, S-2, 1)

        # ── Concatenate all features ──
        features = torch.cat([
            normalized_deficit,     # 1: force inconsistency (the key FCS signal)
            required_force_mag / (self.body_mass * self.gravity),  # 1: normalized required force
            available_force / (self.body_mass * self.gravity),     # 1: normalized available force
            num_contacts,           # 1: number of foot contacts
            foot_skating,           # 1: skating metric
            penetration,            # 1: ground penetration
            com_acc,                # 3: CoM acceleration xyz
            mean_joint_vel,         # 1: mean joint velocity
            max_joint_vel,          # 1: max joint velocity
            mean_joint_acc,         # 1: mean joint acceleration
            max_joint_acc,          # 1: max joint acceleration
            com_height,             # 1: CoM height
        ], dim=-1)  # (B, S-2, 14)

        return features


class FCSPredictor(nn.Module):
    """
    Predicts FCS from joint positions using physics-informed features.

    Two-stage architecture:
    1. PhysicsFeatureExtractor: differentiable physics computation
    2. Small learned head: combines per-frame physics features into scalar FCS
    """

    def __init__(self, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()

        self.physics = PhysicsFeatureExtractor()
        physics_feat_dim = 14  # from PhysicsFeatureExtractor

        # Small temporal network on physics features
        self.temporal = nn.Sequential(
            nn.Linear(physics_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Head: pool over time → scalar
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # FCS is always positive
        )

    def forward(self, joint_positions):
        """
        Args:
            joint_positions: (B, S, 24, 3)

        Returns:
            fcs_pred: (B,) predicted FCS scores
        """
        # Extract physics features
        feats = self.physics(joint_positions)  # (B, S-2, 14)

        # Temporal processing
        x = self.temporal(feats)  # (B, S-2, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, S-2)

        for conv_block in self.conv_blocks:
            x = x + conv_block(x)

        x = x.transpose(1, 2)  # (B, S-2, hidden_dim)

        # Global average pooling
        x = x.mean(dim=1)  # (B, hidden_dim)

        # Predict scalar FCS
        fcs_pred = self.head(x).squeeze(-1)  # (B,)
        return fcs_pred


class FCSPredictorLoss(nn.Module):
    """
    Loss for training FCS predictor.

    Works in log-space to handle the wide range of FCS values (0.02 to 6+).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_fcs, true_fcs):
        """
        Args:
            pred_fcs: (B,) predicted FCS scores (positive, from Softplus)
            true_fcs: (B,) ground-truth FCS scores (positive)

        Returns:
            loss: scalar loss
        """
        log_pred = torch.log(pred_fcs + 1e-6)
        log_true = torch.log(true_fcs + 1e-6)
        log_mse = F.mse_loss(log_pred, log_true)

        smooth_l1 = F.smooth_l1_loss(pred_fcs, true_fcs)

        return log_mse + 0.1 * smooth_l1
