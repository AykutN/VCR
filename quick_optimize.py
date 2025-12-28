#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Optimization - Focused search around promising values
"""

import numpy as np
from batch_test import detect_deepfake, load_reference_samples
import os

def quick_optimize(real_dir='data/real', cloned_dir='data/cloned', max_files=10):
    """Quick optimization focusing on promising parameter ranges."""
    
    print("="*70)
    print("QUICK OPTIMIZATION")
    print("="*70)
    
    # Load reference samples
    print("\nLoading reference samples...")
    all_reference_samples = load_reference_samples(real_dir)
    print(f"Loaded {len(all_reference_samples)} reference samples.\n")
    
    # Get test files
    speaker_dirs = [d for d in os.listdir(real_dir) 
                   if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
    
    test_files = []
    for speaker_dir in speaker_dirs[:1]:
        real_speaker_path = os.path.join(real_dir, speaker_dir)
        cloned_speaker_path = os.path.join(cloned_dir, speaker_dir)
        
        if not os.path.exists(cloned_speaker_path):
            continue
        
        real_files = sorted([f for f in os.listdir(real_speaker_path) if f.endswith('.wav')])
        
        for wav_file in real_files[:max_files]:
            real_path = os.path.join(real_speaker_path, wav_file)
            cloned_path = os.path.join(cloned_speaker_path, wav_file)
            
            if os.path.exists(cloned_path):
                test_files.append({
                    'real': real_path,
                    'cloned': cloned_path,
                    'file': wav_file
                })
    
    print(f"Testing with {len(test_files)} file pairs.\n")
    
    # Focused search space - around promising values
    thresholds = [0.38, 0.40, 0.42, 0.44, 0.46, 0.48]
    
    # Promising weight combinations
    weight_combinations = [
        (0.2, 0.6, 0.2),   # Heavy threshold
        (0.15, 0.7, 0.15), # Very heavy threshold
        (0.1, 0.8, 0.1),   # Extreme threshold
        (0.25, 0.5, 0.25), # Balanced
        (0.2, 0.5, 0.3),   # More threshold
    ]
    
    distance_scales = [6.0, 7.0, 8.0, 9.0, 10.0]
    
    best_score = 0
    best_params = None
    results = []
    
    total = len(thresholds) * len(weight_combinations) * len(distance_scales)
    print(f"Testing {total} combinations (focused search)...\n")
    
    count = 0
    for threshold in thresholds:
        for weights in weight_combinations:
            for dist_scale in distance_scales:
                count += 1
                
                correct_real = 0
                correct_cloned = 0
                total_tests = 0
                
                for test_pair in test_files:
                    try:
                        # Test real
                        result_real = detect_deepfake(
                            test_pair['real'],
                            reference_real_samples=all_reference_samples,
                            threshold=threshold,
                            weights={'distance': weights[0],
                                   'threshold': weights[1],
                                   'statistical': weights[2]},
                            distance_scale=dist_scale
                        )
                        
                        if result_real['is_fake'] is not None:
                            if not result_real['is_fake']:
                                correct_real += 1
                            total_tests += 1
                        
                        # Test cloned
                        result_clone = detect_deepfake(
                            test_pair['cloned'],
                            reference_real_samples=all_reference_samples,
                            threshold=threshold,
                            weights={'distance': weights[0],
                                   'threshold': weights[1],
                                   'statistical': weights[2]},
                            distance_scale=dist_scale
                        )
                        
                        if result_clone['is_fake'] is not None:
                            if result_clone['is_fake']:
                                correct_cloned += 1
                            total_tests += 1
                            
                    except Exception as e:
                        continue
                
                if total_tests > 0:
                    accuracy = (correct_real + correct_cloned) / total_tests
                    real_acc = correct_real / (total_tests / 2) if total_tests > 0 else 0
                    cloned_acc = correct_cloned / (total_tests / 2) if total_tests > 0 else 0
                    
                    results.append({
                        'threshold': threshold,
                        'weights': weights,
                        'distance_scale': dist_scale,
                        'accuracy': accuracy,
                        'real_accuracy': real_acc,
                        'cloned_accuracy': cloned_acc,
                        'correct_real': correct_real,
                        'correct_cloned': correct_cloned
                    })
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_params = {
                            'threshold': threshold,
                            'weights': weights,
                            'distance_scale': dist_scale,
                            'accuracy': accuracy,
                            'real_accuracy': real_acc,
                            'cloned_accuracy': cloned_acc
                        }
                        print(f"✓ New best: {best_score:.2%} | Threshold: {threshold:.2f} | "
                              f"Weights: {weights} | Scale: {dist_scale:.1f}")
    
    # Sort and show top results
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 5 RESULTS")
    print("="*70)
    for i, res in enumerate(results[:5], 1):
        print(f"\n{i}. Accuracy: {res['accuracy']:.2%} | Real: {res['real_accuracy']:.2%} | Cloned: {res['cloned_accuracy']:.2%}")
        print(f"   Threshold: {res['threshold']:.2f} | Weights: {res['weights']} | Scale: {res['distance_scale']:.1f}")
    
    print("\n" + "="*70)
    print("BEST PARAMETERS")
    print("="*70)
    if best_params:
        print(f"\nThreshold: {best_params['threshold']:.2f}")
        print(f"Weights (distance, threshold, statistical): {best_params['weights']}")
        print(f"Distance Scale: {best_params['distance_scale']:.1f}")
        print(f"\nOverall Accuracy: {best_params['accuracy']:.2%}")
        print(f"Real Accuracy: {best_params['real_accuracy']:.2%}")
        print(f"Cloned Accuracy: {best_params['cloned_accuracy']:.2%}")
        
        print("\n" + "="*70)
        print("TEST WITH ALL FILES")
        print("="*70)
        print(f"""
python batch_test.py --threshold {best_params['threshold']:.2f} \\
    --weights "{best_params['weights'][0]:.2f},{best_params['weights'][1]:.2f},{best_params['weights'][2]:.2f}" \\
    --distance-scale {best_params['distance_scale']:.1f}
""")
    
    return best_params, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick parameter optimization')
    parser.add_argument('--real-dir', type=str, default='data/real')
    parser.add_argument('--cloned-dir', type=str, default='data/cloned')
    parser.add_argument('--max-files', type=int, default=10)
    
    args = parser.parse_args()
    
    quick_optimize(
        real_dir=args.real_dir,
        cloned_dir=args.cloned_dir,
        max_files=args.max_files
    )

