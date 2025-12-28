#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Threshold and Weight Optimization

Quick grid search to find optimal parameters.
"""

import numpy as np
from batch_test import batch_test_all_files, detect_deepfake, load_reference_samples
import os

def optimize_parameters(real_dir='data/real', cloned_dir='data/cloned', max_files=10):
    """
    Simple grid search for threshold and weights optimization.
    """
    print("="*70)
    print("SIMPLE PARAMETER OPTIMIZATION")
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
    
    # Search space
    thresholds = np.arange(0.35, 0.56, 0.02)  # 0.35 to 0.55, step 0.02
    
    weight_combinations = [
        (0.2, 0.5, 0.3),   # More threshold
        (0.3, 0.4, 0.3),   # Balanced
        (0.3, 0.5, 0.2),   # Less statistical
        (0.25, 0.5, 0.25), # Balanced 2
        (0.2, 0.6, 0.2),   # Heavy threshold
        (0.15, 0.7, 0.15), # Very heavy threshold
        (0.1, 0.8, 0.1),   # Extreme threshold
    ]
    
    distance_scales = [5.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    
    best_score = 0
    best_params = None
    results = []
    
    total = len(thresholds) * len(weight_combinations) * len(distance_scales)
    print(f"Testing {total} combinations...\n")
    
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
                            if not result_real['is_fake']:  # Should be False for real
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
                            if result_clone['is_fake']:  # Should be True for cloned
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
                
                if count % 20 == 0:
                    print(f"Progress: {count}/{total} ({count/total*100:.1f}%) - Best so far: {best_score:.2%}")
                    if best_params:
                        print(f"  Current best: threshold={best_params['threshold']:.2f}, "
                              f"weights={best_params['weights']}, scale={best_params['distance_scale']:.1f}")
                
                # Early stopping if we found a very good result
                if best_score >= 0.90 and count > 50:
                    print(f"\nFound excellent result ({best_score:.2%}), stopping early...")
                    break
    
    # Sort and show top results
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 RESULTS")
    print("="*70)
    for i, res in enumerate(results[:10], 1):
        print(f"\n{i}. Accuracy: {res['accuracy']:.2%} | Real: {res['real_accuracy']:.2%} | Cloned: {res['cloned_accuracy']:.2%}")
        print(f"   Threshold: {res['threshold']:.2f}")
        print(f"   Weights: {res['weights']}")
        print(f"   Distance Scale: {res['distance_scale']:.1f}")
    
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
        print("USAGE")
        print("="*70)
        print(f"""
# Use in batch_test.py:
python batch_test.py --threshold {best_params['threshold']:.2f}

# Or modify batch_test.py to use:
weights = {best_params['weights']}
distance_scale = {best_params['distance_scale']:.1f}
""")
    
    return best_params, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize detection parameters')
    parser.add_argument('--real-dir', type=str, default='data/real')
    parser.add_argument('--cloned-dir', type=str, default='data/cloned')
    parser.add_argument('--max-files', type=int, default=10)
    
    args = parser.parse_args()
    
    optimize_parameters(
        real_dir=args.real_dir,
        cloned_dir=args.cloned_dir,
        max_files=args.max_files
    )

