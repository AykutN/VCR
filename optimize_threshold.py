#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold and Weight Optimization Script

Grid search to find optimal threshold and feature weights for deepfake detection.
"""

import os
import numpy as np
from batch_test import batch_test_all_files, load_reference_samples, detect_deepfake
import itertools

def grid_search_optimization(real_dir='data/real', cloned_dir='data/cloned', 
                            test_files=None, max_test_files=10):
    """
    Perform grid search to find optimal threshold and weights.
    
    Parameters:
    -----------
    real_dir : str
        Directory containing real audio samples
    cloned_dir : str
        Directory containing cloned audio samples
    test_files : list, optional
        List of specific files to test (for faster search)
    max_test_files : int
        Maximum number of files to use for optimization (for speed)
    """
    
    print("="*70)
    print("GRID SEARCH OPTIMIZATION")
    print("="*70)
    
    # Load reference samples
    print("\nLoading reference samples...")
    all_reference_samples = load_reference_samples(real_dir)
    print(f"Loaded {len(all_reference_samples)} reference samples.\n")
    
    # Get test files
    if test_files is None:
        speaker_dirs = [d for d in os.listdir(real_dir) 
                       if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
        
        test_files = []
        for speaker_dir in speaker_dirs[:1]:  # Use first speaker for optimization
            real_speaker_path = os.path.join(real_dir, speaker_dir)
            cloned_speaker_path = os.path.join(cloned_dir, speaker_dir)
            
            if not os.path.exists(cloned_speaker_path):
                continue
            
            real_files = sorted([f for f in os.listdir(real_speaker_path) if f.endswith('.wav')])
            
            for wav_file in real_files[:max_test_files]:  # Limit for speed
                real_path = os.path.join(real_speaker_path, wav_file)
                cloned_path = os.path.join(cloned_speaker_path, wav_file)
                
                if os.path.exists(cloned_path):
                    test_files.append({
                        'real': real_path,
                        'cloned': cloned_path,
                        'file': wav_file
                    })
    
    print(f"Using {len(test_files)} file pairs for optimization.\n")
    
    # Define search space
    thresholds = [0.35, 0.37, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55]
    
    # Weight combinations (distance, threshold, statistical)
    weight_combinations = [
        (0.2, 0.5, 0.3),   # More weight on threshold
        (0.3, 0.4, 0.3),   # Balanced
        (0.3, 0.5, 0.2),   # Less statistical
        (0.4, 0.4, 0.2),   # More distance
        (0.2, 0.6, 0.2),   # Heavy threshold
        (0.25, 0.5, 0.25), # Balanced 2
        (0.35, 0.35, 0.3), # More balanced
        (0.1, 0.7, 0.2),   # Very heavy threshold
    ]
    
    # Distance scaling factors (for normalization)
    distance_scales = [5.0, 7.0, 10.0, 12.0, 15.0]
    
    best_score = 0
    best_params = None
    results = []
    
    total_combinations = len(thresholds) * len(weight_combinations) * len(distance_scales)
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    count = 0
    for threshold in thresholds:
        for weights in weight_combinations:
            for dist_scale in distance_scales:
                count += 1
                if count % 10 == 0:
                    print(f"Progress: {count}/{total_combinations} combinations tested...")
                
                # Test with these parameters
                correct_real = 0
                correct_cloned = 0
                total = 0
                
                for test_pair in test_files:
                    try:
                        # Test real
                        result_real = detect_deepfake(
                            test_pair['real'], 
                            reference_real_samples=all_reference_samples,
                            threshold=threshold,
                            weights={'distance': weights[0], 
                                   'threshold': weights[1], 
                                   'statistical': weights[2]}
                        )
                        
                        # Modify distance score normalization
                        if result_real['feature_analysis']:
                            dist_score = result_real['feature_analysis']['hybrid_components']['distance']
                            # Recalculate with new scale
                            mean_dist = result_real['feature_analysis']['distance_metrics'].get('mean_euclidean', 0)
                            if mean_dist != np.inf and mean_dist != 0:
                                dist_score = min(mean_dist / dist_scale, 1.0)
                            
                            # Recalculate hybrid score
                            threshold_score = result_real['feature_analysis']['hybrid_components']['threshold']
                            statistical_score = result_real['feature_analysis']['hybrid_components']['statistical']
                            
                            hybrid_score = (
                                weights[0] * dist_score +
                                weights[1] * threshold_score +
                                weights[2] * statistical_score
                            )
                            hybrid_score = max(0.0, min(1.0, hybrid_score))
                            
                            is_fake = hybrid_score >= threshold
                            
                            if not is_fake:  # Real should be False
                                correct_real += 1
                        
                        # Test cloned
                        result_clone = detect_deepfake(
                            test_pair['cloned'],
                            reference_real_samples=all_reference_samples,
                            threshold=threshold,
                            weights={'distance': weights[0],
                                   'threshold': weights[1],
                                   'statistical': weights[2]}
                        )
                        
                        if result_clone['feature_analysis']:
                            dist_score = result_clone['feature_analysis']['hybrid_components']['distance']
                            mean_dist = result_clone['feature_analysis']['distance_metrics'].get('mean_euclidean', 0)
                            if mean_dist != np.inf and mean_dist != 0:
                                dist_score = min(mean_dist / dist_scale, 1.0)
                            
                            threshold_score = result_clone['feature_analysis']['hybrid_components']['threshold']
                            statistical_score = result_clone['feature_analysis']['hybrid_components']['statistical']
                            
                            hybrid_score = (
                                weights[0] * dist_score +
                                weights[1] * threshold_score +
                                weights[2] * statistical_score
                            )
                            hybrid_score = max(0.0, min(1.0, hybrid_score))
                            
                            is_fake = hybrid_score >= threshold
                            
                            if is_fake:  # Cloned should be True
                                correct_cloned += 1
                        
                        total += 2
                        
                    except Exception as e:
                        continue
                
                if total > 0:
                    accuracy = (correct_real + correct_cloned) / total
                    real_acc = correct_real / (total / 2) if total > 0 else 0
                    cloned_acc = correct_cloned / (total / 2) if total > 0 else 0
                    
                    results.append({
                        'threshold': threshold,
                        'weights': weights,
                        'distance_scale': dist_scale,
                        'accuracy': accuracy,
                        'real_accuracy': real_acc,
                        'cloned_accuracy': cloned_acc,
                        'correct_real': correct_real,
                        'correct_cloned': correct_cloned,
                        'total': total
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
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print top 10 results
    print("\n" + "="*70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*70)
    for i, res in enumerate(results[:10], 1):
        print(f"\n{i}. Accuracy: {res['accuracy']:.2%}")
        print(f"   Threshold: {res['threshold']:.2f}")
        print(f"   Weights (distance, threshold, statistical): {res['weights']}")
        print(f"   Distance Scale: {res['distance_scale']:.1f}")
        print(f"   Real Accuracy: {res['real_accuracy']:.2%}")
        print(f"   Cloned Accuracy: {res['cloned_accuracy']:.2%}")
    
    print("\n" + "="*70)
    print("BEST PARAMETERS")
    print("="*70)
    if best_params:
        print(f"\nThreshold: {best_params['threshold']:.2f}")
        print(f"Weights: {best_params['weights']}")
        print(f"Distance Scale: {best_params['distance_scale']:.1f}")
        print(f"Overall Accuracy: {best_params['accuracy']:.2%}")
        print(f"Real Accuracy: {best_params['real_accuracy']:.2%}")
        print(f"Cloned Accuracy: {best_params['cloned_accuracy']:.2%}")
        
        # Generate code snippet
        print("\n" + "="*70)
        print("RECOMMENDED CODE")
        print("="*70)
        print(f"""
# Use these parameters in batch_test.py or detect_deepfake():

threshold = {best_params['threshold']:.2f}
weights = {{
    'distance': {best_params['weights'][0]:.2f},
    'threshold': {best_params['weights'][1]:.2f},
    'statistical': {best_params['weights'][2]:.2f}
}}
distance_scale = {best_params['distance_scale']:.1f}

# Then modify compute_hybrid_score() to use distance_scale:
# distance_score = min(mean_dist / distance_scale, 1.0)
""")
    
    return best_params, results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize threshold and weights for deepfake detection')
    parser.add_argument('--real-dir', type=str, default='data/real',
                       help='Directory containing real audio samples')
    parser.add_argument('--cloned-dir', type=str, default='data/cloned',
                       help='Directory containing cloned audio samples')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum number of files to use for optimization')
    
    args = parser.parse_args()
    
    best_params, all_results = grid_search_optimization(
        real_dir=args.real_dir,
        cloned_dir=args.cloned_dir,
        max_test_files=args.max_files
    )
    
    return best_params, all_results

if __name__ == "__main__":
    main()

