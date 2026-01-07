#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export Predictions for Competition

This script processes test audio files and exports predictions in CSV format
for the competition submission.

Format:
filename,prediction,confidence
file_001.wav,0,0.95
file_002.wav,1,0.82

prediction: 0 = Real, 1 = Fake
"""

import os
import csv
import argparse
from batch_test import detect_deepfake, load_reference_samples

def export_predictions(test_dir, output_file='predictions.csv', 
                      real_dir='data/real', threshold=0.34,
                      weights=None, distance_scale=10.0):
    """
    Process test audio files and export predictions in CSV format.
    
    Parameters:
    -----------
    test_dir : str
        Directory containing test audio files
    output_file : str
        Output CSV file path
    real_dir : str
        Directory containing real audio samples (for reference)
    threshold : float
        Threshold for detection (default: 0.34)
    weights : dict, optional
        Weights for hybrid scoring
    distance_scale : float
        Distance normalization scale
    
    Returns:
    --------
    results : list
        List of prediction results
    """
    print("="*70)
    print("EXPORTING PREDICTIONS")
    print("="*70)
    
    # Load reference samples
    print("\nLoading reference samples...")
    reference_samples = load_reference_samples(real_dir)
    if len(reference_samples) == 0:
        print("⚠ Warning: No reference samples found. Using default detection.")
        reference_samples = None
    else:
        print(f"✓ Loaded {len(reference_samples)} reference samples.")
    
    # Find all audio files in test directory
    print(f"\nScanning test directory: {test_dir}")
    audio_files = []
    
    if os.path.isfile(test_dir):
        # Single file
        if test_dir.endswith('.wav'):
            audio_files.append(('', os.path.basename(test_dir), test_dir))
    elif os.path.isdir(test_dir):
        # Directory - find all .wav files
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.wav'):
                    rel_path = os.path.relpath(root, test_dir)
                    full_path = os.path.join(root, file)
                    audio_files.append((rel_path, file, full_path))
    else:
        print(f"❌ Error: {test_dir} is not a valid file or directory")
        return []
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    # Process each file
    results = []
    print("\nProcessing files...")
    
    for rel_path, filename, filepath in audio_files:
        try:
            result = detect_deepfake(
                filepath,
                reference_real_samples=reference_samples,
                real_dir=real_dir,
                threshold=threshold,
                weights=weights,
                distance_scale=distance_scale
            )
            
            # Convert to competition format
            # prediction: 0 = Real, 1 = Fake
            prediction = 1 if result.get('is_fake') else 0
            confidence = result.get('confidence', 0.0)
            
            # Use relative filename for output
            if rel_path:
                output_filename = f"{rel_path}/{filename}"
            else:
                output_filename = filename
            
            results.append({
                'filename': output_filename,
                'prediction': prediction,
                'confidence': confidence,
                'score': result.get('score', 0.0),
                'is_fake': result.get('is_fake')
            })
            
            print(f"  {filename}: prediction={prediction}, confidence={confidence:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Error processing {filename}: {e}")
            # Still add entry with default values
            if rel_path:
                output_filename = f"{rel_path}/{filename}"
            else:
                output_filename = filename
            results.append({
                'filename': output_filename,
                'prediction': 0,  # Default to Real
                'confidence': 0.0,
                'score': 0.0,
                'is_fake': None
            })
            continue
    
    # Write to CSV
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['filename', 'prediction', 'confidence'])
        
        # Write data
        for result in results:
            writer.writerow([
                result['filename'],
                result['prediction'],
                f"{result['confidence']:.4f}"
            ])
    
    print(f"✓ Exported {len(results)} predictions to {output_file}")
    
    # Print summary
    if results:
        real_count = sum(1 for r in results if r['prediction'] == 0)
        fake_count = sum(1 for r in results if r['prediction'] == 1)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print("\n" + "="*70)
        print("EXPORT SUMMARY")
        print("="*70)
        print(f"Total files: {len(results)}")
        print(f"Predicted as Real (0): {real_count}")
        print(f"Predicted as Fake (1): {fake_count}")
        print(f"Average confidence: {avg_confidence:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Export predictions in CSV format for competition'
    )
    parser.add_argument('test_dir', type=str,
                       help='Directory or file containing test audio files')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path (default: predictions.csv)')
    parser.add_argument('--real-dir', type=str, default='data/real',
                       help='Directory containing real audio samples')
    parser.add_argument('--threshold', type=float, default=0.34,
                       help='Detection threshold (default: 0.34)')
    parser.add_argument('--distance-scale', type=float, default=10.0,
                       help='Distance normalization scale (default: 10.0)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Weights as "distance,threshold,statistical" (e.g., "0.3,0.4,0.3")')
    
    args = parser.parse_args()
    
    weights = None
    if args.weights:
        parts = [float(x.strip()) for x in args.weights.split(',')]
        if len(parts) == 3:
            weights = {
                'distance': parts[0],
                'threshold': parts[1],
                'statistical': parts[2]
            }
    
    export_predictions(
        test_dir=args.test_dir,
        output_file=args.output,
        real_dir=args.real_dir,
        threshold=args.threshold,
        weights=weights,
        distance_scale=args.distance_scale
    )

if __name__ == "__main__":
    main()

