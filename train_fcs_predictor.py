"""
Train FCS Predictor Network on real AIST++ data with robust augmentation.

The predictor must generalize to motions of varying quality — from clean mocap
to noisy diffusion model outputs. We achieve this by training on:
  1. Real mocap motions (low FCS — physically plausible)
  2. Augmented motions with controlled physics violations (high FCS)
     - Joint noise injection (jitter)
     - Foot skating injection (slide grounded feet)
     - Trajectory corruption (unrealistic accelerations)
     - Temporal shuffling (break smoothness)
     - Gravity violation (feet through floor, floating)

Each augmented motion has its FCS recomputed, so the predictor learns the
true mapping from motion quality to FCS score across the full range.

Usage:
    accelerate launch train_fcs_predictor.py --batch_size 32 --epochs 200
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from dataset.quaternion import ax_from_6v
from eval.eval_fcs import ForceConsistencyEvaluator
from model.fcs_predictor import FCSPredictor, FCSPredictorLoss
from vis import SMPLSkeleton


# ─────────────────────────── Augmentations ────────────────────────────


def augment_joint_noise(joints, scale_range=(0.005, 0.05)):
    """Add Gaussian noise to joint positions. Simulates jittery predictions."""
    scale = np.random.uniform(*scale_range)
    noise = np.random.randn(*joints.shape).astype(np.float32) * scale
    return joints + noise


def augment_foot_skating(joints, foot_idx=[7, 8, 10, 11], strength_range=(0.01, 0.08)):
    """Inject horizontal sliding on grounded feet. Simulates foot skating artifacts."""
    joints = joints.copy()
    strength = np.random.uniform(*strength_range)

    min_height = joints[:, foot_idx, 2].min()
    for fi in foot_idx:
        foot_h = joints[:, fi, 2] - min_height
        grounded = foot_h < 0.08  # frames where foot is near ground
        if grounded.sum() < 5:
            continue
        # Add cumulative drift in XY while "grounded"
        drift_dir = np.random.randn(2).astype(np.float32)
        drift_dir /= np.linalg.norm(drift_dir) + 1e-8
        drift = np.zeros_like(joints[:, fi, :2])
        for t in range(1, len(joints)):
            if grounded[t]:
                drift[t] = drift[t - 1] + drift_dir * strength
            else:
                drift[t] = 0.0
        joints[:, fi, :2] += drift
    return joints


def augment_trajectory_corruption(joints, scale_range=(0.02, 0.1)):
    """Add low-frequency noise to root trajectory. Creates unrealistic accelerations."""
    joints = joints.copy()
    S = joints.shape[0]
    scale = np.random.uniform(*scale_range)

    # Low-frequency sinusoidal perturbation on root (joint 0)
    freqs = np.random.uniform(0.5, 3.0, size=3)
    phases = np.random.uniform(0, 2 * np.pi, size=3)
    t = np.linspace(0, 1, S)
    for axis in range(3):
        perturbation = scale * np.sin(2 * np.pi * freqs[axis] * t + phases[axis])
        joints[:, :, axis] += perturbation[:, None]
    return joints


def augment_temporal_jitter(joints, swap_prob=0.15):
    """Randomly swap adjacent frames. Breaks temporal smoothness."""
    joints = joints.copy()
    S = joints.shape[0]
    for t in range(0, S - 1):
        if np.random.random() < swap_prob:
            joints[t], joints[t + 1] = joints[t + 1].copy(), joints[t].copy()
    return joints


def augment_gravity_violation(joints, foot_idx=[7, 8, 10, 11], mode=None):
    """Push feet through floor or make body float. Violates ground contact physics."""
    joints = joints.copy()
    if mode is None:
        mode = np.random.choice(["penetrate", "float"])

    min_height = joints[:, :, 2].min()

    if mode == "penetrate":
        # Push feet below ground plane
        depth = np.random.uniform(0.02, 0.10)
        for fi in foot_idx:
            foot_h = joints[:, fi, 2] - min_height
            low_frames = foot_h < 0.05
            joints[low_frames, fi, 2] -= depth
    else:
        # Lift entire body so feet never touch ground
        lift = np.random.uniform(0.1, 0.4)
        joints[:, :, 2] += lift
    return joints


def augment_limb_explosion(joints):
    """Scale random limb segments outward. Creates impossible poses."""
    joints = joints.copy()
    # Pick a random limb chain
    limb_chains = [
        [16, 18, 20, 22],  # left arm
        [17, 19, 21, 23],  # right arm
        [1, 4, 7, 10],     # left leg
        [2, 5, 8, 11],     # right leg
    ]
    chain = limb_chains[np.random.randint(len(limb_chains))]
    scale = np.random.uniform(1.3, 2.5)

    # Scale the limb relative to its root
    root_joint = chain[0]
    anchor = joints[:, root_joint:root_joint + 1, :]  # (S, 1, 3)
    for ji in chain[1:]:
        offset = joints[:, ji, :] - anchor[:, 0, :]
        joints[:, ji, :] = anchor[:, 0, :] + offset * scale
    return joints


def augment_velocity_spike(joints, num_spikes_range=(2, 8)):
    """Insert sudden velocity spikes at random frames. Creates force inconsistencies."""
    joints = joints.copy()
    S = joints.shape[0]
    num_spikes = np.random.randint(*num_spikes_range)

    for _ in range(num_spikes):
        t = np.random.randint(1, S - 1)
        spike = np.random.randn(joints.shape[1], 3).astype(np.float32)
        magnitude = np.random.uniform(0.05, 0.3)
        joints[t] += spike * magnitude
    return joints


# All augmentation functions with their probability of being applied
AUGMENTATIONS = [
    (augment_joint_noise, 1.0),
    (augment_foot_skating, 0.7),
    (augment_trajectory_corruption, 0.6),
    (augment_temporal_jitter, 0.5),
    (augment_gravity_violation, 0.6),
    (augment_limb_explosion, 0.4),
    (augment_velocity_spike, 0.5),
]


def apply_random_augmentations(joints, min_augs=1, max_augs=4):
    """Apply a random subset of augmentations to create a corrupted motion."""
    joints = joints.copy()

    # Shuffle augmentation order
    aug_order = list(range(len(AUGMENTATIONS)))
    np.random.shuffle(aug_order)

    num_to_apply = np.random.randint(min_augs, max_augs + 1)
    applied = 0

    for idx in aug_order:
        if applied >= num_to_apply:
            break
        aug_fn, prob = AUGMENTATIONS[idx]
        if np.random.random() < prob:
            joints = aug_fn(joints)
            applied += 1

    return joints


# ─────────────────────────── Dataset ────────────────────────────


class FCSDataset(Dataset):
    """Dataset of (joint_positions, fcs_score) pairs."""

    def __init__(self, joint_positions_list, fcs_scores_list):
        self.joint_positions = joint_positions_list
        self.fcs_scores = fcs_scores_list

    def __len__(self):
        return len(self.fcs_scores)

    def __getitem__(self, idx):
        joints = torch.from_numpy(self.joint_positions[idx]).float()
        fcs = torch.tensor(self.fcs_scores[idx]).float()
        return joints, fcs


class AugmentedFCSDataset(Dataset):
    """
    Dataset that augments real motions on-the-fly to produce diverse FCS training pairs.
    Each epoch generates fresh augmentations, so the predictor sees new corruptions every time.
    """

    def __init__(self, real_joints_list, real_fcs_list, evaluator, aug_ratio=3):
        """
        Args:
            real_joints_list: List of (S, 24, 3) numpy arrays — real mocap
            real_fcs_list: List of floats — ground truth FCS for real data
            evaluator: ForceConsistencyEvaluator instance
            aug_ratio: Number of augmented samples per real sample
        """
        self.real_joints = real_joints_list
        self.real_fcs = real_fcs_list
        self.evaluator = evaluator
        self.aug_ratio = aug_ratio
        self.n_real = len(real_joints_list)

        # Pre-generate initial augmented data
        self._regenerate_augmented()

    def _regenerate_augmented(self):
        """Generate augmented samples with computed FCS. Called each epoch."""
        self.aug_joints = []
        self.aug_fcs = []

        for i in range(self.n_real):
            for _ in range(self.aug_ratio):
                aug = apply_random_augmentations(self.real_joints[i])
                try:
                    result = self.evaluator.evaluate_motion(aug)
                    self.aug_joints.append(aug)
                    self.aug_fcs.append(result['fcs_score'])
                except Exception:
                    continue

        # Combine real + augmented
        self.all_joints = self.real_joints + self.aug_joints
        self.all_fcs = self.real_fcs + self.aug_fcs

    def __len__(self):
        return len(self.all_fcs)

    def __getitem__(self, idx):
        joints = torch.from_numpy(self.all_joints[idx]).float()
        fcs = torch.tensor(self.all_fcs[idx]).float()
        return joints, fcs


# ─────────────────────────── Data Loading ────────────────────────────


def extract_joint_positions(dataset, smpl, device, max_samples=500):
    """Extract joint positions and compute FCS from AIST++ dataset."""
    evaluator = ForceConsistencyEvaluator(fps=30)

    joint_positions_list = []
    fcs_scores_list = []

    num_samples = min(max_samples, len(dataset))
    print(f"Processing {num_samples} sequences...")

    for idx in tqdm(range(num_samples)):
        try:
            motion_data, _, _, _ = dataset[idx]
            motion_data = motion_data.unsqueeze(0)
            motion_data = dataset.normalizer.unnormalize(motion_data)
            motion_data = motion_data.squeeze(0)

            seq_len = motion_data.shape[0]
            root_pos = motion_data[:, 4:7]
            local_q_6d = motion_data[:, 7:].reshape(seq_len, 24, 6)
            local_q = ax_from_6v(local_q_6d)

            joint_positions = smpl.forward(
                local_q.unsqueeze(0).to(device),
                root_pos.unsqueeze(0).to(device),
            )
            joints_np = joint_positions.squeeze(0).cpu().numpy()

            result = evaluator.evaluate_motion(joints_np)
            joint_positions_list.append(joints_np)
            fcs_scores_list.append(result['fcs_score'])

        except Exception as e:
            print(f"\nError processing sequence {idx}: {e}")
            continue

    print(f"Processed {len(fcs_scores_list)} sequences")
    fcs_arr = np.array(fcs_scores_list)
    print(f"FCS range: [{fcs_arr.min():.4f}, {fcs_arr.max():.4f}], mean: {fcs_arr.mean():.4f}")

    return joint_positions_list, fcs_scores_list


# ─────────────────────────── Training ────────────────────────────


def train_predictor(train_dataset, val_dataset, args, accelerator):
    """Train FCS predictor network."""

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    model = FCSPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    print(f"FCS Predictor: {sum(p.numel() for p in model.parameters())} parameters")

    criterion = FCSPredictorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Regenerate augmented data every N epochs for fresh corruptions
        if hasattr(train_dataset, '_regenerate_augmented') and epoch % args.regen_interval == 0 and epoch > 1:
            if accelerator.is_main_process:
                print(f"  [Epoch {epoch}] Regenerating augmented training data...")
            train_dataset._regenerate_augmented()
            # Need to re-create the dataloader after dataset changes
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
            )
            train_loader = accelerator.prepare(train_loader)

        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        for joints, fcs_true in train_loader:
            optimizer.zero_grad()
            fcs_pred = model(joints)
            loss = criterion(fcs_pred, fcs_true)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for joints, fcs_true in val_loader:
                fcs_pred = model(joints)
                loss = criterion(fcs_pred, fcs_true)
                val_loss += loss.item()
                val_batches += 1
                val_preds.extend(fcs_pred.cpu().tolist())
                val_trues.extend(fcs_true.cpu().tolist())

        val_loss /= max(val_batches, 1)
        scheduler.step()

        # Correlation metric (how well does it rank motions)
        if len(val_preds) > 2:
            val_preds_arr = np.array(val_preds)
            val_trues_arr = np.array(val_trues)
            correlation = np.corrcoef(val_preds_arr, val_trues_arr)[0, 1]
        else:
            correlation = 0.0

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Corr: {correlation:.4f} | "
                f"Data: {len(train_dataset)} samples"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'correlation': correlation,
                    'args': args,
                }, args.save_path)
                if epoch % 10 == 0:
                    print(f"  -> Saved best model (val_loss: {val_loss:.4f}, corr: {correlation:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/dataset_backups/train_tensor_dataset.pkl')
    parser.add_argument('--test_data', type=str, default='data/dataset_backups/test_tensor_dataset.pkl')
    parser.add_argument('--max_train_samples', type=int, default=1000)
    parser.add_argument('--max_val_samples', type=int, default=186)
    parser.add_argument('--save_path', type=str, default='models/fcs_predictor.pt')
    parser.add_argument('--aug_ratio', type=int, default=3,
                        help='Number of augmented versions per real sample')
    parser.add_argument('--regen_interval', type=int, default=20,
                        help='Regenerate augmented data every N epochs')

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience')

    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load SMPL skeleton
    smpl = SMPLSkeleton(device)
    evaluator = ForceConsistencyEvaluator(fps=30)

    # ── Prepare training data ──
    if accelerator.is_main_process:
        print("=" * 70)
        print("PREPARING TRAINING DATA")
        print("=" * 70)

    train_raw = pickle.load(open(args.train_data, 'rb'))
    train_joints, train_fcs = extract_joint_positions(
        train_raw, smpl, device, args.max_train_samples
    )

    if accelerator.is_main_process:
        print(f"\nCreating augmented dataset (ratio={args.aug_ratio})...")
        print(f"Expected size: {len(train_joints)} real + ~{len(train_joints) * args.aug_ratio} augmented")

    train_dataset = AugmentedFCSDataset(
        train_joints, train_fcs, evaluator, aug_ratio=args.aug_ratio
    )

    if accelerator.is_main_process:
        fcs_all = np.array(train_dataset.all_fcs)
        print(f"Training dataset: {len(train_dataset)} total samples")
        print(f"  Real:      {len(train_joints)} (FCS range: [{min(train_fcs):.4f}, {max(train_fcs):.4f}])")
        print(f"  Augmented: {len(train_dataset.aug_fcs)} (FCS range: [{min(train_dataset.aug_fcs):.4f}, {max(train_dataset.aug_fcs):.4f}])")
        print(f"  Combined FCS: mean={fcs_all.mean():.4f}, std={fcs_all.std():.4f}")

    # ── Prepare validation data (real only, no augmentation) ──
    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("PREPARING VALIDATION DATA")
        print("=" * 70)

    test_raw = pickle.load(open(args.test_data, 'rb'))
    val_joints, val_fcs = extract_joint_positions(
        test_raw, smpl, device, args.max_val_samples
    )

    # Validation set: real + one round of augmentation (for coverage)
    val_dataset = AugmentedFCSDataset(
        val_joints, val_fcs, evaluator, aug_ratio=1
    )

    if accelerator.is_main_process:
        print(f"Validation dataset: {len(val_dataset)} samples")

    accelerator.wait_for_everyone()

    # ── Train ──
    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("TRAINING FCS PREDICTOR")
        print("=" * 70)

    model = train_predictor(train_dataset, val_dataset, args, accelerator)

    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("DONE!")
        print("=" * 70)
        print(f"Use this predictor in training with:")
        print(f"  accelerate launch train.py --fcs_predictor_path {args.save_path} --fcs_loss_weight 0.1")


if __name__ == '__main__':
    main()
