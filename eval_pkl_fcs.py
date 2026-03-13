"""Evaluate FCS on saved motion .pkl files"""
import argparse
import glob
import os
import pickle
import numpy as np
from tqdm import tqdm
from eval.eval_fcs import ForceConsistencyEvaluator, calculate_pfc_score


def evaluate_pkl_directory(motion_dir):
    """Evaluate FCS and PFC on directory of .pkl files"""
    pkl_files = glob.glob(os.path.join(motion_dir, "*.pkl"))
    
    if len(pkl_files) == 0:
        print(f"No .pkl files found in {motion_dir}")
        return
    
    print(f"Found {len(pkl_files)} motion files")
    
    # Initialize evaluator
    fcs_evaluator = ForceConsistencyEvaluator(fps=30)
    
    fcs_scores = []
    pfc_scores = []
    contact_ratios = []
    
    for pkl_file in tqdm(pkl_files, desc="Evaluating physics"):
        # Load motion
        data = pickle.load(open(pkl_file, "rb"))
        joint_positions = data["full_pose"]  # (T, 24, 3)
        
        # Calculate FCS
        fcs_result = fcs_evaluator.evaluate_motion(joint_positions)
        fcs_scores.append(fcs_result['fcs_score'])
        contact_ratios.append(fcs_result['contact_ratio'])
        
        # Calculate PFC
        pfc_score = calculate_pfc_score(joint_positions)
        pfc_scores.append(pfc_score)
    
    # Statistics
    fcs_scores = np.array(fcs_scores)
    pfc_scores = np.array(pfc_scores)
    contact_ratios = np.array(contact_ratios)
    
    # Filter invalid
    valid_fcs = fcs_scores[~np.isnan(fcs_scores) & ~np.isinf(fcs_scores)]
    valid_pfc = pfc_scores[~np.isnan(pfc_scores) & ~np.isinf(pfc_scores)]
    
    print("\n" + "="*70)
    print(f"Physics Evaluation: {motion_dir}")
    print("="*70)
    print(f"Total samples:        {len(pkl_files)}")
    print(f"Valid FCS:            {len(valid_fcs)}")
    print(f"Valid PFC:            {len(valid_pfc)}")
    print("-"*70)
    print(f"FCS Score (mean):     {valid_fcs.mean():.3f}")
    print(f"FCS Score (std):      {valid_fcs.std():.3f}")
    print(f"FCS Score (median):   {np.median(valid_fcs):.3f}")
    print("-"*70)
    print(f"PFC Score (mean):     {valid_pfc.mean():.3f}")
    print(f"PFC Score (std):      {valid_pfc.std():.3f}")
    print(f"PFC Score (median):   {np.median(valid_pfc):.3f}")
    print("-"*70)
    print(f"Contact ratio (mean): {contact_ratios.mean()*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="eval/generated_motions",
        help="Directory containing motion .pkl files"
    )
    args = parser.parse_args()
    
    evaluate_pkl_directory(args.motion_path)
