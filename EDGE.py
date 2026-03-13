import multiprocessing
import os
import pickle
import csv
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from dataset.quaternion import ax_from_6v
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        fcs_loss_weight=0.0,
        fcs_predictor_path="",
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"
        
        # Store FCS loss weight
        self.fcs_loss_weight = fcs_loss_weight
        self.fcs_predictor = None

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device, weights_only=False
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)
        
        # Initialize FCS evaluator for physics validation
        try:
            from eval.eval_fcs import ForceConsistencyEvaluator, calculate_pfc_score
            self.fcs_evaluator = ForceConsistencyEvaluator(fps=30)
            self.calculate_pfc = calculate_pfc_score
            self.use_fcs = True
            print("="*60)
            print("✓ FCS evaluator initialized for physics monitoring")
            print("✓ PFC metric available for comparison")
            if self.fcs_loss_weight > 0:
                print(f"✓ FCS loss enabled with weight: {self.fcs_loss_weight}")
            print("="*60)
        except Exception as e:
            self.use_fcs = False
            print("="*60)
            print(f"✗ FCS evaluator initialization failed: {type(e).__name__}: {e}")
            print("✗ FCS/PFC physics validation will be skipped")
            print("="*60)
        
        # Load FCS predictor network if provided
        if fcs_predictor_path != "":
            try:
                from model.fcs_predictor import FCSPredictor
                checkpoint_fcs = torch.load(fcs_predictor_path, map_location=self.accelerator.device, weights_only=False)
                predictor_args = checkpoint_fcs.get('args', None)
                
                # Create predictor with same architecture as training
                if predictor_args:
                    self.fcs_predictor = FCSPredictor(
                        hidden_dim=predictor_args.hidden_dim,
                        num_layers=predictor_args.num_layers,
                        dropout=predictor_args.dropout
                    ).to(self.accelerator.device)
                else:
                    # Default architecture
                    self.fcs_predictor = FCSPredictor().to(self.accelerator.device)
                
                self.fcs_predictor.load_state_dict(checkpoint_fcs['model_state_dict'])
                self.fcs_predictor.eval()  # Always in eval mode
                
                print("="*60)
                print(f"✓ FCS predictor loaded from {fcs_predictor_path}")
                print(f"✓ Differentiable physics loss enabled")
                print("="*60)
            except Exception as e:
                self.fcs_predictor = None
                print("="*60)
                print(f"✗ Failed to load FCS predictor: {e}")
                print(f"✗ Falling back to monitoring-only mode")
                print("="*60)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def evaluate_fcs_on_batch(self, cond, num_samples=4):
        """
        Evaluate Force Consistency Score and PFC on generated samples.
        
        Args:
            cond: Conditioning features (music) - (B, S, C)
            num_samples: Number of samples to evaluate
        
        Returns:
            dict with mean_fcs_score, mean_pfc_score and individual scores
        """
        if not self.use_fcs:
            print("[Physics] Skipped - evaluator not initialized")
            return {'mean_fcs_score': 0.0, 'mean_pfc_score': 0.0, 'individual_fcs': [], 'individual_pfc': [], 'num_evaluated': 0}
        
        self.eval()
        
        try:
            # Generate samples
            batch_size = min(num_samples, len(cond))
            shape = (batch_size, self.horizon, self.repr_dim)
            
            print(f"[FCS] Generating {batch_size} samples for evaluation...")
            
            with torch.no_grad():
                # Sample from the diffusion model using DDIM sampling
                samples = self.diffusion.ddim_sample(shape, cond[:batch_size])
                
                # Unnormalize samples (FCS needs real-world units)
                samples = self.normalizer.unnormalize(samples)
                
                # Convert to joint positions for FCS evaluation
                b, s, c = samples.shape
                print(f"[FCS] Samples shape: {samples.shape}")
                
                # Split off contact labels (first 4 channels)
                sample_contact, samples = torch.split(samples, (4, c - 4), dim=2)
                
                # Extract position and rotation
                sample_x = samples[:, :, :3]  # (b, s, 3)
                sample_q = ax_from_6v(samples[:, :, 3:].reshape(b, s, 24, 6))  # (b, s, 24, 3)
                
                print(f"[FCS] Running forward kinematics...")
                # Forward kinematics to get joint positions
                joint_positions = self.diffusion.smpl.forward(sample_q, sample_x)  # (b, s, 24, 3)
                print(f"[FCS] Joint positions shape: {joint_positions.shape}")
                
                # Evaluate FCS and PFC for each sample
                fcs_scores = []
                pfc_scores = []
                for i in range(batch_size):
                    joints_np = joint_positions[i].cpu().numpy()  # (s, 24, 3)
                    
                    try:
                        # Calculate FCS
                        fcs_result = self.fcs_evaluator.evaluate_motion(joints_np)
                        fcs_scores.append(fcs_result['fcs_score'])
                        
                        # Calculate PFC
                        pfc_score = self.calculate_pfc(joints_np)
                        pfc_scores.append(pfc_score)
                        
                        print(f"[Physics] Sample {i+1}/{batch_size}: FCS={fcs_result['fcs_score']:.4f}, PFC={pfc_score:.4f}")
                    except Exception as e:
                        print(f"[Physics] ERROR evaluating sample {i+1}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if len(fcs_scores) == 0:
                    print("[Physics] WARNING: All samples failed evaluation!")
                    return {'mean_fcs_score': 0.0, 'mean_pfc_score': 0.0, 'individual_fcs': [], 'individual_pfc': [], 'num_evaluated': 0}
                
                mean_fcs = float(sum(fcs_scores) / len(fcs_scores))
                mean_pfc = float(sum(pfc_scores) / len(pfc_scores))
                print(f"[Physics] Mean FCS: {mean_fcs:.4f}, Mean PFC: {mean_pfc:.4f} (from {len(fcs_scores)}/{batch_size} samples)")
                
        except Exception as e:
            print(f"[Physics] CRITICAL ERROR in evaluation pipeline: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {'mean_fcs_score': 0.0, 'mean_pfc_score': 0.0, 'individual_fcs': [], 'individual_pfc': [], 'num_evaluated': 0}
        finally:
            self.train()
        
        return {
            'mean_fcs_score': mean_fcs,
            'mean_pfc_score': mean_pfc,
            'individual_fcs': fcs_scores,
            'individual_pfc': pfc_scores,
            'num_evaluated': len(fcs_scores)
        }
    
    def compute_physics_penalty(self, cond, num_samples=4):
        """
        Compute a simplified, differentiable physics penalty on generated samples.
        This is a lightweight approximation of FCS for use during training.
        
        Penalizes:
        - Feet going through ground (height < 0)
        - High accelerations when feet should be in contact
        - Unrealistic velocities
        
        Args:
            cond: Conditioning features (B, S, C)
            num_samples: Number of samples to generate
        
        Returns:
            penalty: Scalar differentiable tensor
            stats: Dict with penalty components for logging
        """
        if self.fcs_loss_weight <= 0:
            return torch.tensor(0.0, device=cond.device), {}
        
        # Generate clean samples (with gradients!)
        batch_size = min(num_samples, len(cond))
        shape = (batch_size, self.horizon, self.repr_dim)
        
        # Sample with gradients enabled
        samples = self.diffusion.ddim_sample(shape, cond[:batch_size])
        
        # Unnormalize
        samples = self.normalizer.unnormalize(samples)
        
        # Parse motion: contacts(4), root(3), rotations(24*6)
        b, s, c = samples.shape
        sample_contact, motion = torch.split(samples, (4, c - 4), dim=2)
        sample_x = motion[:, :, :3]  # root position
        sample_q = ax_from_6v(motion[:, :, 3:].reshape(b, s, 24, 6))  # rotations
        
        # Forward kinematics to get joint positions
        joint_positions = self.diffusion.smpl.forward(sample_q, sample_x)  # (b, s, 24, 3)
        
        # Get foot joint indices (ankles and toes)
        foot_idx = [7, 8, 10, 11]  # L_Ankle, R_Ankle, L_Toe, R_Toe
        foot_positions = joint_positions[:, :, foot_idx, :]  # (b, s, 4, 3)
        
        # Penalty 1: Feet going through ground (height < 0)
        # Adjust heights relative to minimum (ground level)
        min_height = foot_positions[:, :, :, 2].min(dim=1, keepdim=True)[0]  # (b, 1, 4)
        adjusted_heights = foot_positions[:, :, :, 2] - min_height  # (b, s, 4)
        ground_penetration = torch.relu(-adjusted_heights)  # penalize negative heights
        ground_penalty = ground_penetration.mean()
        
        # Penalty 2: Foot skating (high velocity when near ground)
        # NOTE: Reduced weight since controlled sliding is necessary in dance (pivots, turns)
        foot_velocity = torch.norm(foot_positions[:, 1:] - foot_positions[:, :-1], dim=-1)  # (b, s-1, 4)
        near_ground = (adjusted_heights[:, :-1] < 0.08)  # feet within 8cm of ground
        skating_penalty = (foot_velocity * near_ground.float()).mean()
        
        # Penalty 3: Excessive accelerations (unrealistic forces)
        foot_accel = foot_velocity[:, 1:] - foot_velocity[:, :-1]  # (b, s-2, 4)
        accel_penalty = torch.relu(foot_accel.abs() - 2.0).mean()  # penalize > 2 m/s² change
        
        # Combine penalties (prioritize ground penetration, allow some skating for dance)
        total_penalty = (
            2.0 * ground_penalty +      # CRITICAL: feet must not go through floor
            0.1 * skating_penalty +     # LOW: controlled sliding needed for pivots/turns
            0.3 * accel_penalty         # MODERATE: limit unrealistic accelerations
        )
        
        stats = {
            'ground_penalty': float(ground_penalty),
            'skating_penalty': float(skating_penalty),
            'accel_penalty': float(accel_penalty),
            'total_penalty': float(total_penalty)
        }
        
        return total_penalty, stats
    
    def compute_fcs_loss_with_predictor(self, x, cond):
        """
        Compute differentiable FCS loss using the predictor network.
        This works on the CURRENT DENOISING STEP outputs, not final samples.
        
        Args:
            x: (B, S, repr_dim) current motion representation
            cond: (B, S, C) conditioning features
            
        Returns:
            fcs_loss: Scalar differentiable tensor
            mean_fcs: Float for logging
        """
        if self.fcs_predictor is None:
            return torch.tensor(0.0, device=x.device), 0.0
        
        # Parse motion (already unnormalized in training)
        b, s, c = x.shape
        
        # Split: contacts(4), root(3), rotations(24*6)
        contact, motion = torch.split(x, (4, c - 4), dim=2)
        root_pos = motion[:, :, :3]
        rotations_6d = motion[:, :, 3:].reshape(b, s, 24, 6)
        rotations = ax_from_6v(rotations_6d)  # (b, s, 24, 3)
        
        # Forward kinematics to get joint positions
        joint_positions = self.diffusion.smpl.forward(rotations, root_pos)  # (b, s, 24, 3)
        
        # Predict FCS using trained network
        with torch.set_grad_enabled(True):
            fcs_pred = self.fcs_predictor(joint_positions)  # (b,)
        
        # Loss is mean predicted FCS (want to minimize it)
        fcs_loss = fcs_pred.mean()
        mean_fcs = float(fcs_loss.detach().cpu().item())
        
        return fcs_loss, mean_fcs

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            
            # Initialize metrics logging files
            metrics_file = save_dir / "training_metrics.csv"
            metrics_json_file = save_dir / "training_metrics.json"
            
            # Create CSV file with headers
            with open(metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Timestamp', 'Total_Loss', 'Train_Loss', 
                                'V_Loss', 'FK_Loss', 'Foot_Loss', 'FCS_Train_Loss', 'FCS_Score', 'PFC_Score', 'Type'])
            
            # Initialize JSON metrics list
            all_metrics = []
            
            print(f"  Metrics will be saved to:")
            print(f"    - {metrics_file}")
            print(f"    - {metrics_json_file}")

        self.accelerator.wait_for_everyone()
        
        # Print training configuration
        if self.accelerator.is_main_process:
            print(f"\n{'='*70}")
            print(f"STARTING TRAINING")
            print(f"{'='*70}")
            print(f"  Experiment:      {opt.exp_name}")
            print(f"  Total Epochs:    {opt.epochs}")
            print(f"  Batch Size:      {opt.batch_size}")
            print(f"  Save Interval:   {opt.save_interval} epochs")
            print(f"  Feature Type:    {opt.feature_type}")
            print(f"  FCS Monitoring:  {self.use_fcs}")
            if self.fcs_loss_weight > 0:
                print(f" Physics Regularization:")
                print(f"    Weight:       {self.fcs_loss_weight}")
                print(f"    Every:        {opt.fcs_regularize_every} epochs")
                print(f"    Samples:      {opt.fcs_num_samples}")
            print(f"  Progress prints: Every 50 epochs")
            print(f"{'='*70}\n")
        
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_fcs_loss = 0
            fcs_steps = 0
            
            # train
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                
                # Add FCS loss if predictor is available
                # Apply to every Nth batch to balance compute cost
                if self.fcs_predictor is not None and step % opt.fcs_regularize_every == 0:
                    # Unnormalize x to get real-world units for FCS
                    x_unnorm = self.normalizer.unnormalize(x)
                    fcs_loss, mean_fcs = self.compute_fcs_loss_with_predictor(x_unnorm, cond)
                    total_loss = total_loss + self.fcs_loss_weight * fcs_loss
                    
                    if self.accelerator.is_main_process:
                        avg_fcs_loss += mean_fcs
                        fcs_steps += 1
                
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            
            # Periodic physics evaluation (if enabled)
            # Note: This monitors physics quality but doesn't apply gradients
            # Full backprop through DDIM sampling is not feasible
            if (self.fcs_loss_weight > 0 and 
                epoch % opt.fcs_regularize_every == 0 and
                self.accelerator.is_main_process):
                
                print(f"\n[Physics] Monitoring physics quality at epoch {epoch}...")
                
                # Get a batch for evaluation
                eval_iter = iter(train_data_loader)
                x_eval, cond_eval, _, _ = next(eval_iter)
                
                try:
                    # Compute physics metrics (no gradient)
                    self.eval()
                    with torch.no_grad():
                        physics_penalty, penalty_stats = self.compute_physics_penalty(
                            cond_eval, 
                            num_samples=opt.fcs_num_samples
                        )
                    
                    # Log physics penalties
                    print(f"[Physics] Metrics - "
                          f"Ground: {penalty_stats.get('ground_penalty', 0):.4f}, "
                          f"Skating: {penalty_stats.get('skating_penalty', 0):.4f}, "
                          f"Accel: {penalty_stats.get('accel_penalty', 0):.4f}, "
                          f"Total: {penalty_stats.get('total_penalty', 0):.4f}")
                    
                    # Log to wandb if available
                    if wandb.run is not None:
                        wandb.log({
                            'physics/ground_penalty': penalty_stats.get('ground_penalty', 0),
                            'physics/skating_penalty': penalty_stats.get('skating_penalty', 0),  
                            'physics/accel_penalty': penalty_stats.get('accel_penalty', 0),
                            'physics/total_penalty': penalty_stats.get('total_penalty', 0),
                            'epoch': epoch
                        })
                    
                    self.train()  # back to train mode
                    
                except Exception as e:
                    print(f"[Physics] Error during monitoring: {e}")
                    import traceback
                    traceback.print_exc()
                    self.train()
            
            # Print progress every 50 epochs
            if self.accelerator.is_main_process and (epoch % 50 == 0):
                temp_avg_loss = avg_loss / len(train_data_loader)
                temp_avg_vloss = avg_vloss / len(train_data_loader)
                temp_avg_fkloss = avg_fkloss / len(train_data_loader)
                temp_avg_footloss = avg_footloss / len(train_data_loader)
                total = temp_avg_loss + temp_avg_vloss + temp_avg_fkloss + temp_avg_footloss
                
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{opt.epochs} - Training Progress")
                print(f"{'='*70}")
                print(f"  Total Loss:      {total:.6f}")
                print(f"  Reconstruction:  {temp_avg_loss:.6f}")
                print(f"  Velocity Loss:   {temp_avg_vloss:.6f}")
                print(f"  FK Loss:         {temp_avg_fkloss:.6f}")
                print(f"  Foot Loss:       {temp_avg_footloss:.6f}")
                if fcs_steps > 0:
                    temp_avg_fcs = avg_fcs_loss / fcs_steps
                    print(f"  FCS Loss (pred): {temp_avg_fcs:.6f}")
                print(f"{'='*70}\n")
                print(f"  ├─ Train Loss:   {temp_avg_loss:.6f}  ({temp_avg_loss/total*100:5.1f}%)")
                print(f"  ├─ V Loss:       {temp_avg_vloss:.6f}  ({temp_avg_vloss/total*100:5.1f}%)")
                print(f"  ├─ FK Loss:      {temp_avg_fkloss:.6f}  ({temp_avg_fkloss/total*100:5.1f}%)")
                print(f"  └─ Foot Loss:    {temp_avg_footloss:.6f}  ({temp_avg_footloss/total*100:5.1f}%)")
                print(f"{'='*70}\n")
                
                # Save progress metrics to files
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                temp_avg_fcs = avg_fcs_loss / fcs_steps if fcs_steps > 0 else 0.0
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, timestamp, f"{total:.6f}", f"{temp_avg_loss:.6f}",
                                   f"{temp_avg_vloss:.6f}", f"{temp_avg_fkloss:.6f}", 
                                   f"{temp_avg_footloss:.6f}", f"{temp_avg_fcs:.6f}" if fcs_steps > 0 else "N/A",
                                   "N/A", "N/A", "progress"])
                
                all_metrics.append({
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'total_loss': float(total),
                    'train_loss': float(temp_avg_loss),
                    'v_loss': float(temp_avg_vloss),
                    'fk_loss': float(temp_avg_fkloss),
                    'foot_loss': float(temp_avg_footloss),
                    'fcs_train_loss': float(temp_avg_fcs) if fcs_steps > 0 else None,
                    'fcs_score': None,
                    'pfc_score': None,
                    'type': 'progress'
                })
            
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    
                    # Evaluate FCS and PFC on validation samples
                    physics_result = {'mean_fcs_score': 0.0, 'mean_pfc_score': 0.0}
                    if self.use_fcs:
                        try:
                            # Use test data for physics evaluation
                            (_, val_cond, _, _) = next(iter(test_data_loader))
                            val_cond = val_cond.to(self.accelerator.device)
                            print("[Physics] Evaluating FCS and PFC quality...")
                            physics_result = self.evaluate_fcs_on_batch(val_cond, num_samples=4)
                            print(f"[Physics] FCS: {physics_result['mean_fcs_score']:.4f}, "
                                  f"PFC: {physics_result['mean_pfc_score']:.4f} "
                                  f"({physics_result['num_evaluated']} samples evaluated)")
                        except Exception as e:
                            import traceback
                            print(f"[Physics] Evaluation failed: {type(e).__name__}: {e}")
                            print(traceback.format_exc())
                    else:
                        print("[Physics] Skipped - evaluator not initialized")
                    
                    # Print detailed checkpoint summary
                    total = avg_loss + avg_vloss + avg_fkloss + avg_footloss
                    print(f"\n{'#'*70}")
                    print(f"# CHECKPOINT - Epoch {epoch}/{opt.epochs}")
                    print(f"{'#'*70}")
                    print(f"  Total Loss:      {total:.6f}")
                    print(f"  ├─ Train Loss:   {avg_loss:.6f}")
                    print(f"  ├─ V Loss:       {avg_vloss:.6f}")
                    print(f"  ├─ FK Loss:      {avg_fkloss:.6f}")
                    print(f"  ├─ Foot Loss:    {avg_footloss:.6f}")
                    print(f"  ├─ FCS Score:    {physics_result['mean_fcs_score']:.6f}")
                    print(f"  └─ PFC Score:    {physics_result['mean_pfc_score']:.6f}")
                    print(f"  Model saved to:  weights/train-{epoch}.pt")
                    print(f"{'#'*70}\n")
                    
                    # Save checkpoint metrics to files
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    temp_avg_fcs = avg_fcs_loss / fcs_steps if fcs_steps > 0 else 0.0
                    with open(metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, timestamp, f"{total:.6f}", f"{avg_loss:.6f}",
                                       f"{avg_vloss:.6f}", f"{avg_fkloss:.6f}", 
                                       f"{avg_footloss:.6f}", f"{temp_avg_fcs:.6f}" if fcs_steps > 0 else "N/A",
                                       f"{physics_result['mean_fcs_score']:.6f}",
                                       f"{physics_result['mean_pfc_score']:.6f}", "checkpoint"])
                    
                    all_metrics.append({
                        'epoch': epoch,
                        'timestamp': timestamp,
                        'total_loss': float(total),
                        'train_loss': float(avg_loss),
                        'v_loss': float(avg_vloss),
                        'fk_loss': float(avg_fkloss),
                        'foot_loss': float(avg_footloss),
                        'fcs_train_loss': float(temp_avg_fcs) if fcs_steps > 0 else None,
                        'fcs_score': float(physics_result['mean_fcs_score']),
                        'pfc_score': float(physics_result['mean_pfc_score']),
                        'type': 'checkpoint'
                    })
                    
                    # Save JSON file (updated each checkpoint)
                    with open(metrics_json_file, 'w') as f:
                        json.dump({
                            'experiment': opt.exp_name,
                            'total_epochs': opt.epochs,
                            'batch_size': opt.batch_size,
                            'feature_type': opt.feature_type,
                            'metrics': all_metrics
                        }, f, indent=2)
                    
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "FCS Score": physics_result['mean_fcs_score'],
                        "PFC Score": physics_result['mean_pfc_score'],
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        
        # Training completion summary
        if self.accelerator.is_main_process:
            # Final save of metrics
            with open(metrics_json_file, 'w') as f:
                json.dump({
                    'experiment': opt.exp_name,
                    'total_epochs': opt.epochs,
                    'batch_size': opt.batch_size,
                    'feature_type': opt.feature_type,
                    'completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'metrics': all_metrics
                }, f, indent=2)
            
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"  Total Epochs:    {opt.epochs}")
            print(f"  Experiment:      {opt.exp_name}")
            print(f"  Final Model:     weights/train-{epoch}.pt")
            print(f"  Metrics saved:")
            print(f"    - {metrics_file}")
            print(f"    - {metrics_json_file}")
            print(f"  Total records:   {len(all_metrics)}")
            print(f"{'='*70}\n")
            print(f"  WandB Project:   {opt.wandb_pj_name}")
            print(f"{'='*70}\n")
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
