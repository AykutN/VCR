#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze scores to understand the distribution and find optimal threshold
"""

import numpy as np
from batch_test import detect_deepfake, load_reference_samples
import os
import matplotlib.pyplot as plt

def analyze_score_distribution(real_dir='data/real', cloned_dir='data/cloned'):
    """Analyze score distributions for real and cloned audio."""
    
    print("="*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load reference samples
    print("\nLoading reference samples...")
    all_reference_samples = load_reference_samples(real_dir)
    print(f"Loaded {len(all_reference_samples)} reference samples.\n")
    
    # Get all files
    speaker_dirs = [d for d in os.listdir(real_dir) 
                   if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
    
    real_scores = []
    cloned_scores = []
    
    for speaker_dir in speaker_dirs:
        real_speaker_path = os.path.join(real_dir, speaker_dir)
        cloned_speaker_path = os.path.join(cloned_dir, speaker_dir)
        
        if not os.path.exists(cloned_speaker_path):
            continue
        
        real_files = sorted([f for f in os.listdir(real_speaker_path) if f.endswith('.wav')])
        
        for wav_file in real_files:
            real_path = os.path.join(real_speaker_path, wav_file)
            cloned_path = os.path.join(cloned_speaker_path, wav_file)
            
            if not os.path.exists(cloned_path):
                continue
            
            try:
                # Test real
                result_real = detect_deepfake(
                    real_path,
                    reference_real_samples=all_reference_samples,
                    threshold=0.5
                )
                if result_real['score'] is not None:
                    real_scores.append(result_real['score'])
                
                # Test cloned
                result_clone = detect_deepfake(
                    cloned_path,
                    reference_real_samples=all_reference_samples,
                    threshold=0.5
                )
                if result_clone['score'] is not None:
                    cloned_scores.append(result_clone['score'])
                    
            except Exception as e:
                continue
    
    real_scores = np.array(real_scores)
    cloned_scores = np.array(cloned_scores)
    
    print("\n" + "="*70)
    print("SCORE STATISTICS")
    print("="*70)
    
    print(f"\nREAL AUDIO:")
    print(f"  Count: {len(real_scores)}")
    print(f"  Mean: {np.mean(real_scores):.4f}")
    print(f"  Std: {np.std(real_scores):.4f}")
    print(f"  Min: {np.min(real_scores):.4f}")
    print(f"  Max: {np.max(real_scores):.4f}")
    print(f"  Median: {np.median(real_scores):.4f}")
    print(f"  25th percentile: {np.percentile(real_scores, 25):.4f}")
    print(f"  75th percentile: {np.percentile(real_scores, 75):.4f}")
    
    print(f"\nCLONED AUDIO:")
    print(f"  Count: {len(cloned_scores)}")
    print(f"  Mean: {np.mean(cloned_scores):.4f}")
    print(f"  Std: {np.std(cloned_scores):.4f}")
    print(f"  Min: {np.min(cloned_scores):.4f}")
    print(f"  Max: {np.max(cloned_scores):.4f}")
    print(f"  Median: {np.median(cloned_scores):.4f}")
    print(f"  25th percentile: {np.percentile(cloned_scores, 25):.4f}")
    print(f"  75th percentile: {np.percentile(cloned_scores, 75):.4f}")
    
    # Find optimal threshold
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*70)
    
    # Test different thresholds
    thresholds = np.arange(0.20, 0.50, 0.01)
    best_threshold = 0.5
    best_accuracy = 0
    
    print("\nTesting thresholds:")
    for threshold in thresholds:
        real_correct = np.sum(real_scores < threshold)
        cloned_correct = np.sum(cloned_scores >= threshold)
        total_correct = real_correct + cloned_correct
        accuracy = total_correct / (len(real_scores) + len(cloned_scores))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
        
        if threshold in [0.25, 0.30, 0.35, 0.40, 0.45]:
            print(f"  Threshold {threshold:.2f}: Accuracy = {accuracy:.2%} "
                  f"(Real: {real_correct}/{len(real_scores)}, Cloned: {cloned_correct}/{len(cloned_scores)})")
    
    print(f"\n✓ Best threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2%}")
    
    # Calculate accuracy with best threshold
    real_correct = np.sum(real_scores < best_threshold)
    cloned_correct = np.sum(cloned_scores >= best_threshold)
    
    print(f"\nWith threshold {best_threshold:.2f}:")
    print(f"  Real accuracy: {real_correct}/{len(real_scores)} ({real_correct/len(real_scores):.2%})")
    print(f"  Cloned accuracy: {cloned_correct}/{len(cloned_scores)} ({cloned_correct/len(cloned_scores):.2%})")
    print(f"  Overall accuracy: {(real_correct + cloned_correct)/(len(real_scores) + len(cloned_scores)):.2%}")
    
    # Separation analysis
    print("\n" + "="*70)
    print("SEPARATION ANALYSIS")
    print("="*70)
    
    overlap = np.sum((real_scores >= np.min(cloned_scores)) & (real_scores <= np.max(cloned_scores)))
    print(f"Overlap: {overlap}/{len(real_scores)} real scores fall within cloned score range")
    print(f"Separation: {(np.mean(cloned_scores) - np.mean(real_scores)):.4f}")
    print(f"Separation (std units): {(np.mean(cloned_scores) - np.mean(real_scores)) / np.std(np.concatenate([real_scores, cloned_scores])):.4f}")
    
    if np.mean(cloned_scores) <= np.mean(real_scores):
        print("\n⚠ WARNING: Cloned scores are NOT higher than real scores on average!")
        print("   This suggests the features are not distinguishing well.")
        print("   Consider:")
        print("   1. Improving feature extraction")
        print("   2. Using different features")
        print("   3. Adjusting distance normalization")
    
    return {
        'real_scores': real_scores,
        'cloned_scores': cloned_scores,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy
    }

if __name__ == "__main__":
    analyze_score_distribution()

