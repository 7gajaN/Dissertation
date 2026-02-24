"""
Integration guide for using FCS during EDGE training.

This script shows how to add FCS validation to the training loop
without modifying the core loss functions.
"""

import os
import torch
import numpy as np
from eval.eval_fcs import ForceConsistencyEvaluator
from model.utils import ax_from_6v


def add_fcs_validation_to_training():
    """
    Modify EDGE.py to add FCS validation during training.
    
    Add this code to the training loop in EDGE.train_loop() method.
    """
    
    example_code = '''
    # Add to EDGE.__init__() after line 100:
    self.fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    
    # Add to EDGE.train_loop() after saving model checkpoint (around line 250):
    if (epoch % opt.save_interval) == 0:
        if self.accelerator.is_main_process:
            # ... existing code ...
            
            # === ADD FCS VALIDATION ===
            print("Evaluating FCS on validation samples...")
            fcs_score = self.evaluate_fcs_on_batch(x, cond)
            log_dict["FCS Score"] = fcs_score
            print(f"Epoch {epoch} - FCS Score: {fcs_score:.4f}")
            # === END FCS VALIDATION ===
            
            wandb.log(log_dict)
    '''
    
    return example_code


def evaluate_fcs_on_batch_method():
    """
    Add this method to the EDGE class to evaluate FCS.
    """
    
    method_code = '''
    def evaluate_fcs_on_batch(self, x, cond, num_samples=4):
        """
        Evaluate FCS on a batch of generated samples.
        
        Args:
            x: Ground truth motions (not used, just for batch size)
            cond: Conditioning features (music)
            num_samples: Number of samples to evaluate
        
        Returns:
            mean_fcs_score: Average FCS score across samples
        """
        self.eval()
        
        # Generate samples
        batch_size = min(num_samples, len(cond))
        shape = (batch_size, self.horizon, self.repr_dim)
        
        with torch.no_grad():
            # Sample from the diffusion model
            samples = self.diffusion.sample(shape, cond[:batch_size])
            
            # Convert to joint positions for FCS evaluation
            b, s, c = samples.shape
            
            # Extract position and rotation
            sample_x = samples[:, :, :3]  # (b, s, 3)
            sample_q = ax_from_6v(samples[:, :, 7:].reshape(b, s, 24, 6))  # (b, s, 24, 3)
            
            # Forward kinematics to get joint positions
            joint_positions = self.diffusion.smpl.forward(sample_q, sample_x)  # (b, s, 24, 3)
            
            # Evaluate FCS for each sample
            fcs_scores = []
            for i in range(batch_size):
                joints_np = joint_positions[i].cpu().numpy()  # (s, 24, 3)
                
                try:
                    result = self.fcs_evaluator.evaluate_motion(joints_np)
                    fcs_scores.append(result['fcs_score'])
                except Exception as e:
                    print(f"FCS evaluation failed for sample {i}: {e}")
                    continue
            
            if len(fcs_scores) == 0:
                return 0.0
            
            mean_fcs = np.mean(fcs_scores)
            
        self.train()
        return mean_fcs
    '''
    
    return method_code


if __name__ == "__main__":
    print("="*60)
    print("FCS Integration Guide for EDGE Training")
    print("="*60)
    print("\n1. ADD FCS VALIDATION TO TRAINING LOOP:")
    print(add_fcs_validation_to_training())
    print("\n2. ADD FCS EVALUATION METHOD TO EDGE CLASS:")
    print(add_fcs_validation_to_batch_method())
    print("\n" + "="*60)
