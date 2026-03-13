"""
Train FCS Predictor Network on real AIST++ data.

This script:
1. Loads real mocap sequences from AIST++ 
2. Computes ground-truth FCS for each
3. Trains predictor network to match FCS scores
4. Saves trained predictor for use in physics-aware training

Usage:
    accelerate launch train_fcs_predictor.py --batch_size 32 --epochs 100
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


class FCSDataset(Dataset):
    """Dataset of (joint_positions, fcs_score) pairs"""
    
    def __init__(self, joint_positions_list, fcs_scores_list):
        self.joint_positions = joint_positions_list
        self.fcs_scores = fcs_scores_list
        
    def __len__(self):
        return len(self.fcs_scores)
    
    def __getitem__(self, idx):
        joints = torch.from_numpy(self.joint_positions[idx]).float()
        fcs = torch.tensor(self.fcs_scores[idx]).float()
        return joints, fcs


def collate_fn(batch):
    """Custom collate to handle variable-length sequences"""
    joints_list, fcs_list = zip(*batch)
    
    # Find max sequence length
    max_len = max(j.shape[0] for j in joints_list)
    
    # Pad sequences
    padded_joints = []
    for joints in joints_list:
        if joints.shape[0] < max_len:
            padding = torch.zeros(max_len - joints.shape[0], 24, 3)
            joints = torch.cat([joints, padding], dim=0)
        padded_joints.append(joints)
    
    joints_batch = torch.stack(padded_joints, dim=0)
    fcs_batch = torch.stack(fcs_list, dim=0)
    
    return joints_batch, fcs_batch


def prepare_dataset(dataset_path, max_samples=500, accelerator=None):
    """Load dataset and compute FCS for all sequences"""
    
    device = accelerator.device if accelerator else 'cpu'
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = pickle.load(open(dataset_path, 'rb'))
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Initialize (on correct device)
    fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    smpl = SMPLSkeleton(device)
    
    joint_positions_list = []
    fcs_scores_list = []
    
    num_samples = min(max_samples, len(dataset))
    print(f"\nComputing FCS for {num_samples} sequences...")
    
    for idx in tqdm(range(num_samples)):
        try:
            # Get motion data
            motion_data, _, _, _ = dataset[idx]
            motion_data = motion_data.unsqueeze(0)
            motion_data = dataset.normalizer.unnormalize(motion_data)
            motion_data = motion_data.squeeze(0)
            
            # Parse motion: contacts(4), root(3), rotations(24*6)
            seq_len, features = motion_data.shape
            contact = motion_data[:, :4]
            root_pos = motion_data[:, 4:7]
            local_q_6d = motion_data[:, 7:]
            
            # Convert to axis-angle
            local_q_6d = local_q_6d.reshape(seq_len, 24, 6)
            local_q = ax_from_6v(local_q_6d)
            
            # Forward kinematics (move to device first)
            local_q = local_q.to(device)
            root_pos = root_pos.to(device)
            joint_positions = smpl.forward(local_q.unsqueeze(0), root_pos.unsqueeze(0))
            joint_positions = joint_positions.squeeze(0).cpu().numpy()
            
            # Compute FCS
            fcs_result = fcs_evaluator.evaluate_motion(joint_positions)
            fcs_score = fcs_result['fcs_score']
            
            joint_positions_list.append(joint_positions)
            fcs_scores_list.append(fcs_score)
            
        except Exception as e:
            print(f"\nError processing sequence {idx}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(fcs_scores_list)} sequences")
    print(f"FCS range: [{min(fcs_scores_list):.4f}, {max(fcs_scores_list):.4f}]")
    print(f"FCS mean: {np.mean(fcs_scores_list):.4f}")
    
    return FCSDataset(joint_positions_list, fcs_scores_list)


def train_predictor(train_dataset, val_dataset, args, accelerator):
    """Train FCS predictor network"""
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    model = FCSPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"\nFCS Predictor: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = FCSPredictorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Prepare with accelerate (handles device placement automatically)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for joints, fcs_true in train_loader:
            # No need for .to(device) - accelerate handles it
            
            optimizer.zero_grad()
            fcs_pred = model(joints)
            loss = criterion(fcs_pred, fcs_true)
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for joints, fcs_true in val_loader:
                # No need for .to(device) - accelerate handles it
                
                fcs_pred = model(joints)
                loss = criterion(fcs_pred, fcs_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model (unwrap model for saving)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }, args.save_path)
                if epoch % 10 == 0:
                    print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/dataset_backups/train_tensor_dataset.pkl')
    parser.add_argument('--test_data', type=str, default='data/dataset_backups/test_tensor_dataset.pkl')
    parser.add_argument('--max_train_samples', type=int, default=400)
    parser.add_argument('--max_val_samples', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='models/fcs_predictor.pt')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        # Create save directory
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare datasets (only on main process to avoid conflicts)
        print("="*70)
        print("PREPARING TRAINING DATA")
        print("="*70)
        train_dataset = prepare_dataset(args.train_data, args.max_train_samples, accelerator)
        
        print("\n" + "="*70)
        print("PREPARING VALIDATION DATA")
        print("="*70)
        val_dataset = prepare_dataset(args.test_data, args.max_val_samples, accelerator)
    else:
        # Other processes just need the data structure, but can reuse from main process
        train_dataset = prepare_dataset(args.train_data, args.max_train_samples, accelerator)
        val_dataset = prepare_dataset(args.test_data, args.max_val_samples, accelerator)
    
    accelerator.wait_for_everyone()
    
    # Train predictor
    if accelerator.is_main_process:
        print("\n" + "="*70)
        print("TRAINING FCS PREDICTOR")
        print("="*70)
    
    model = train_predictor(train_dataset, val_dataset, args, accelerator)
    
    if accelerator.is_main_process:
        print("\n" + "="*70)
        print("DONE!")
        print("="*70)
        print(f"Use this predictor in training with:")
        print(f"  --fcs_predictor_path {args.save_path}")
        print(f"\nOr with accelerate:")
        print(f"  accelerate launch train.py --fcs_predictor_path {args.save_path} --fcs_loss_weight 0.5")


if __name__ == '__main__':
    main()
