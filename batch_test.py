#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Test Script for Deepfake Voice Detection

This script tests all real and cloned audio files and provides detailed statistics.
"""

import os
import sys
import numpy as np
import librosa
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import functions from analysis notebook
# Note: You may need to import these from a module if they're in a separate file
# For now, we'll include the essential functions here

def extract_mfcc(audio, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048):
    """Extract MFCC features from audio signal."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    return mfcc

def extract_delta_features(mfcc):
    """Extract delta and delta-delta features."""
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    return delta, delta_delta

def extract_fourier_features(audio, sr=22050, hop_length=512, n_fft=2048):
    """Extract Fourier transform-based features."""
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio, hop_length=hop_length, frame_length=n_fft
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_bandwidth': spectral_bandwidth,
        'mel_spectrogram': mel_spectrogram
    }

def compute_statistical_features(feature_matrix):
    """Compute statistical features from a feature matrix."""
    stats_dict = {
        'mean': np.mean(feature_matrix, axis=1),
        'std': np.std(feature_matrix, axis=1),
        'min': np.min(feature_matrix, axis=1),
        'max': np.max(feature_matrix, axis=1),
        'median': np.median(feature_matrix, axis=1)
    }
    
    skewness = []
    kurtosis = []
    for i in range(feature_matrix.shape[0]):
        skewness.append(stats.skew(feature_matrix[i, :]))
        kurtosis.append(stats.kurtosis(feature_matrix[i, :]))
    
    stats_dict['skewness'] = np.array(skewness)
    stats_dict['kurtosis'] = np.array(kurtosis)
    
    return stats_dict

def extract_all_features(audio, sr=22050):
    """Extract all features (MFCC, Delta, Delta-Delta, Fourier) from audio."""
    mfcc = extract_mfcc(audio, sr)
    delta, delta_delta = extract_delta_features(mfcc)
    fourier = extract_fourier_features(audio, sr)
    
    features = {
        'mfcc': mfcc,
        'delta': delta,
        'delta_delta': delta_delta,
        'fourier': fourier,
        'mfcc_stats': compute_statistical_features(mfcc),
        'delta_stats': compute_statistical_features(delta),
        'delta_delta_stats': compute_statistical_features(delta_delta)
    }
    
    for key, value in fourier.items():
        if value.ndim == 2:
            features[f'{key}_stats'] = compute_statistical_features(value)
    
    return features

def flatten_statistical_features(features):
    """Flatten statistical features into a single feature vector."""
    feature_list = []
    
    for stat_name in ['mean', 'std', 'skewness', 'kurtosis']:
        if 'mfcc_stats' in features and stat_name in features['mfcc_stats']:
            feature_list.append(features['mfcc_stats'][stat_name])
        if 'delta_stats' in features and stat_name in features['delta_stats']:
            feature_list.append(features['delta_stats'][stat_name])
        if 'delta_delta_stats' in features and stat_name in features['delta_delta_stats']:
            feature_list.append(features['delta_delta_stats'][stat_name])
    
    fourier_keys = ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate', 
                    'spectral_bandwidth']
    for key in fourier_keys:
        stat_key = f'{key}_stats'
        if stat_key in features:
            for stat_name in ['mean', 'std', 'skewness', 'kurtosis']:
                if stat_name in features[stat_key]:
                    feature_list.append(features[stat_key][stat_name])
    
    if len(feature_list) == 0:
        return np.array([])
    
    return np.concatenate(feature_list)

def compute_distance_metrics(test_features, reference_features_list):
    """Compute distance metrics between test features and reference features."""
    test_vector = flatten_statistical_features(test_features)
    
    if len(test_vector) == 0:
        return {'mean_euclidean': np.inf, 'min_euclidean': np.inf, 
                'max_euclidean': np.inf, 'euclidean_distances': []}
    
    euclidean_distances = []
    
    for ref_features in reference_features_list:
        ref_vector = flatten_statistical_features(ref_features)
        
        if len(ref_vector) == 0 or len(ref_vector) != len(test_vector):
            continue
        
        scaler = StandardScaler()
        combined = np.vstack([test_vector.reshape(1, -1), ref_vector.reshape(1, -1)])
        combined_scaled = scaler.fit_transform(combined)
        test_scaled = combined_scaled[0]
        ref_scaled = combined_scaled[1]
        
        dist = euclidean(test_scaled, ref_scaled)
        euclidean_distances.append(dist)
    
    if len(euclidean_distances) == 0:
        return {'mean_euclidean': np.inf, 'min_euclidean': np.inf, 
                'max_euclidean': np.inf, 'euclidean_distances': []}
    
    return {
        'euclidean_distances': euclidean_distances,
        'mean_euclidean': np.mean(euclidean_distances),
        'min_euclidean': np.min(euclidean_distances),
        'max_euclidean': np.max(euclidean_distances),
        'std_euclidean': np.std(euclidean_distances)
    }

def compute_feature_thresholds(reference_features_list):
    """Compute thresholds for each feature based on reference samples."""
    if len(reference_features_list) == 0:
        return {}
    
    all_vectors = []
    for ref_features in reference_features_list:
        vector = flatten_statistical_features(ref_features)
        if len(vector) > 0:
            all_vectors.append(vector)
    
    if len(all_vectors) == 0:
        return {}
    
    all_vectors = np.array(all_vectors)
    mean_features = np.mean(all_vectors, axis=0)
    std_features = np.std(all_vectors, axis=0)
    
    thresholds = {
        'mean': mean_features,
        'std': std_features,
        'lower_bound': mean_features - 2 * std_features,
        'upper_bound': mean_features + 2 * std_features,
        'lower_bound_3sigma': mean_features - 3 * std_features,
        'upper_bound_3sigma': mean_features + 3 * std_features
    }
    
    return thresholds

def threshold_detection(test_features, thresholds):
    """Perform threshold-based detection."""
    test_vector = flatten_statistical_features(test_features)
    
    if len(test_vector) == 0 or 'lower_bound' not in thresholds:
        return {
            'threshold_score': 0.5,
            'features_outside_bounds': 0,
            'features_outside_3sigma': 0,
            'feature_violations': []
        }
    
    if len(test_vector) != len(thresholds['lower_bound']):
        return {
            'threshold_score': 0.5,
            'features_outside_bounds': 0,
            'features_outside_3sigma': 0,
            'feature_violations': []
        }
    
    outside_2sigma = np.sum(
        (test_vector < thresholds['lower_bound']) | 
        (test_vector > thresholds['upper_bound'])
    )
    
    outside_3sigma = np.sum(
        (test_vector < thresholds['lower_bound_3sigma']) | 
        (test_vector > thresholds['upper_bound_3sigma'])
    )
    
    total_features = len(test_vector)
    threshold_score = (outside_2sigma / total_features) * 0.7 + (outside_3sigma / total_features) * 0.3
    
    return {
        'threshold_score': min(threshold_score, 1.0),
        'features_outside_bounds': int(outside_2sigma),
        'features_outside_3sigma': int(outside_3sigma),
        'total_features': total_features,
        'violation_ratio': outside_2sigma / total_features if total_features > 0 else 0
    }

def statistical_comparison(test_features, reference_features_list):
    """Compare test features statistically with reference features."""
    test_vector = flatten_statistical_features(test_features)
    
    if len(test_vector) == 0:
        return {'statistical_score': 0.5}
    
    ref_vectors = []
    for ref_features in reference_features_list:
        vector = flatten_statistical_features(ref_features)
        if len(vector) == len(test_vector):
            ref_vectors.append(vector)
    
    if len(ref_vectors) == 0:
        return {'statistical_score': 0.5}
    
    ref_vectors = np.array(ref_vectors)
    ref_mean = np.mean(ref_vectors, axis=0)
    ref_std = np.std(ref_vectors, axis=0)
    
    z_scores = np.abs((test_vector - ref_mean) / (ref_std + 1e-10))
    
    high_z_score_count = np.sum(z_scores > 2.0)
    very_high_z_score_count = np.sum(z_scores > 3.0)
    
    total_features = len(test_vector)
    statistical_score = (
        (high_z_score_count / total_features) * 0.6 + 
        (very_high_z_score_count / total_features) * 0.4
    )
    
    return {
        'statistical_score': min(statistical_score, 1.0),
        'mean_z_score': np.mean(z_scores),
        'max_z_score': np.max(z_scores),
        'high_z_score_count': int(high_z_score_count),
        'very_high_z_score_count': int(very_high_z_score_count)
    }

def compute_hybrid_score(distance_metrics, threshold_result, statistical_result, 
                         weights=None, distance_scale=10.0):
    """Compute hybrid deepfake score combining multiple detection methods."""
    if weights is None:
        weights = {
            'distance': 0.3,
            'threshold': 0.4,
            'statistical': 0.3
        }
    
    mean_dist = distance_metrics.get('mean_euclidean', 0)
    if mean_dist == np.inf or mean_dist == 0:
        distance_score = 0.5
    else:
        distance_score = min(mean_dist / distance_scale, 1.0)
    
    threshold_score = threshold_result.get('threshold_score', 0.5)
    statistical_score = statistical_result.get('statistical_score', 0.5)
    
    hybrid_score = (
        weights['distance'] * distance_score +
        weights['threshold'] * threshold_score +
        weights['statistical'] * statistical_score
    )
    
    hybrid_score = max(0.0, min(1.0, hybrid_score))
    
    return {
        'hybrid_score': hybrid_score,
        'distance_score': distance_score,
        'threshold_score': threshold_score,
        'statistical_score': statistical_score,
        'components': {
            'distance': distance_score,
            'threshold': threshold_score,
            'statistical': statistical_score
        }
    }

def load_reference_samples(real_dir, max_samples=None):
    """Load reference (real) audio samples and extract their features."""
    reference_features_list = []
    
    if not os.path.exists(real_dir):
        print(f"Warning: Real directory {real_dir} does not exist.")
        return reference_features_list
    
    speaker_dirs = [d for d in os.listdir(real_dir) 
                   if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
    
    sample_count = 0
    for speaker_dir in speaker_dirs:
        speaker_path = os.path.join(real_dir, speaker_dir)
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            if max_samples is not None and sample_count >= max_samples:
                break
            
            wav_path = os.path.join(speaker_path, wav_file)
            try:
                audio, sr = librosa.load(wav_path, sr=22050)
                features = extract_all_features(audio, sr)
                reference_features_list.append(features)
                sample_count += 1
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
        
        if max_samples is not None and sample_count >= max_samples:
            break
    
    print(f"Loaded {len(reference_features_list)} reference samples.")
    return reference_features_list

def detect_deepfake(audio_path, reference_real_samples=None, real_dir=None, 
                   threshold=0.5, weights=None, distance_scale=10.0):
    """Detect if an audio file is a deepfake (cloned) or real."""
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        return {
            'is_fake': None,
            'score': None,
            'confidence': 0.0,
            'error': str(e),
            'feature_analysis': {}
        }
    
    test_features = extract_all_features(audio, sr)
    
    if reference_real_samples is None:
        if real_dir is None:
            real_dir = 'data/real'
        reference_real_samples = load_reference_samples(real_dir)
    
    if len(reference_real_samples) == 0:
        return {
            'is_fake': None,
            'score': 0.5,
            'confidence': 0.0,
            'error': 'No reference samples available',
            'feature_analysis': {}
        }
    
    distance_metrics = compute_distance_metrics(test_features, reference_real_samples)
    thresholds = compute_feature_thresholds(reference_real_samples)
    threshold_result = threshold_detection(test_features, thresholds)
    statistical_result = statistical_comparison(test_features, reference_real_samples)
    hybrid_result = compute_hybrid_score(
        distance_metrics, threshold_result, statistical_result, weights, distance_scale
    )
    
    score = hybrid_result['hybrid_score']
    is_fake = score >= threshold
    
    confidence = abs(score - threshold) * 2
    confidence = min(1.0, confidence)
    
    return {
        'is_fake': is_fake,
        'score': score,
        'confidence': confidence,
        'threshold': threshold,
        'feature_analysis': {
            'distance_metrics': distance_metrics,
            'threshold_result': threshold_result,
            'statistical_result': statistical_result,
            'hybrid_components': hybrid_result['components']
        }
    }

def batch_test_all_files(real_dir='data/real', cloned_dir='data/cloned', threshold=0.5,
                        weights=None, distance_scale=10.0):
    """
    Test all real and cloned audio files and return results.
    
    Parameters:
    -----------
    real_dir : str
        Directory containing real audio samples
    cloned_dir : str
        Directory containing cloned audio samples
    threshold : float
        Threshold for classification
    
    Returns:
    --------
    results : dict
        Dictionary containing test results for all files
    """
    results = {
        'real': [],
        'cloned': [],
        'summary': {}
    }
    
    print("Loading reference samples...")
    all_reference_samples = load_reference_samples(real_dir)
    print(f"Loaded {len(all_reference_samples)} reference samples.\n")
    
    speaker_dirs = [d for d in os.listdir(real_dir) 
                    if os.path.isdir(os.path.join(real_dir, d)) and not d.startswith('.')]
    
    for speaker_dir in speaker_dirs:
        real_speaker_path = os.path.join(real_dir, speaker_dir)
        cloned_speaker_path = os.path.join(cloned_dir, speaker_dir)
        
        if not os.path.exists(cloned_speaker_path):
            print(f"[!] Cloned directory not found for {speaker_dir}, skipping...")
            continue
        
        real_files = sorted([f for f in os.listdir(real_speaker_path) if f.endswith('.wav')])
        
        print(f"\n{'='*70}")
        print(f"Testing speaker: {speaker_dir}")
        print(f"{'='*70}")
        
        for wav_file in real_files:
            real_path = os.path.join(real_speaker_path, wav_file)
            cloned_path = os.path.join(cloned_speaker_path, wav_file)
            
            if not os.path.exists(cloned_path):
                print(f"[!] Cloned file not found: {wav_file}, skipping...")
                continue
            
            try:
                ref_samples = all_reference_samples
                
                result_real = detect_deepfake(real_path, reference_real_samples=ref_samples, 
                                             threshold=threshold, weights=weights,
                                             distance_scale=distance_scale)
                results['real'].append({
                    'file': wav_file,
                    'speaker': speaker_dir,
                    'path': real_path,
                    'is_fake': result_real['is_fake'],
                    'score': result_real['score'],
                    'confidence': result_real['confidence']
                })
                
                result_clone = detect_deepfake(cloned_path, reference_real_samples=ref_samples, 
                                              threshold=threshold, weights=weights,
                                              distance_scale=distance_scale)
                results['cloned'].append({
                    'file': wav_file,
                    'speaker': speaker_dir,
                    'path': cloned_path,
                    'is_fake': result_clone['is_fake'],
                    'score': result_clone['score'],
                    'confidence': result_clone['confidence']
                })
                
                print(f"\nFile: {wav_file}")
                print(f"  Real:   Score={result_real['score']:.4f}, IsFake={result_real['is_fake']}, Conf={result_real['confidence']:.4f}")
                print(f"  Cloned: Score={result_clone['score']:.4f}, IsFake={result_clone['is_fake']}, Conf={result_clone['confidence']:.4f}")
                
            except Exception as e:
                print(f"[!] Error testing {wav_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate summary statistics
    if results['real']:
        real_scores = [r['score'] for r in results['real']]
        results['summary']['real'] = {
            'count': len(results['real']),
            'mean_score': np.mean(real_scores),
            'std_score': np.std(real_scores),
            'min_score': np.min(real_scores),
            'max_score': np.max(real_scores),
            'correct_predictions': sum(1 for r in results['real'] if not r['is_fake']),
            'accuracy': sum(1 for r in results['real'] if not r['is_fake']) / len(results['real'])
        }
    
    if results['cloned']:
        cloned_scores = [r['score'] for r in results['cloned']]
        results['summary']['cloned'] = {
            'count': len(results['cloned']),
            'mean_score': np.mean(cloned_scores),
            'std_score': np.std(cloned_scores),
            'min_score': np.min(cloned_scores),
            'max_score': np.max(cloned_scores),
            'correct_predictions': sum(1 for r in results['cloned'] if r['is_fake']),
            'accuracy': sum(1 for r in results['cloned'] if r['is_fake']) / len(results['cloned'])
        }
    
    return results

def print_summary(batch_results):
    """Print summary statistics from batch test results."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    if 'real' in batch_results['summary']:
        real_sum = batch_results['summary']['real']
        print(f"\nREAL AUDIO:")
        print(f"  Total files: {real_sum['count']}")
        print(f"  Mean score: {real_sum['mean_score']:.4f} (std: {real_sum['std_score']:.4f})")
        print(f"  Score range: [{real_sum['min_score']:.4f}, {real_sum['max_score']:.4f}]")
        print(f"  Correct predictions: {real_sum['correct_predictions']}/{real_sum['count']}")
        print(f"  Accuracy: {real_sum['accuracy']:.2%}")

    if 'cloned' in batch_results['summary']:
        clone_sum = batch_results['summary']['cloned']
        print(f"\nCLONED AUDIO:")
        print(f"  Total files: {clone_sum['count']}")
        print(f"  Mean score: {clone_sum['mean_score']:.4f} (std: {clone_sum['std_score']:.4f})")
        print(f"  Score range: [{clone_sum['min_score']:.4f}, {clone_sum['max_score']:.4f}]")
        print(f"  Correct predictions: {clone_sum['correct_predictions']}/{clone_sum['count']}")
        print(f"  Accuracy: {clone_sum['accuracy']:.2%}")

    # Overall accuracy
    if 'real' in batch_results['summary'] and 'cloned' in batch_results['summary']:
        total_correct = (batch_results['summary']['real']['correct_predictions'] + 
                        batch_results['summary']['cloned']['correct_predictions'])
        total_files = (batch_results['summary']['real']['count'] + 
                      batch_results['summary']['cloned']['count'])
        overall_accuracy = total_correct / total_files if total_files > 0 else 0
        print(f"\nOVERALL ACCURACY: {overall_accuracy:.2%} ({total_correct}/{total_files})")

    print("\n" + "="*70)

def main():
    """Main function to run batch test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test deepfake voice detection')
    parser.add_argument('--real-dir', type=str, default='data/real',
                       help='Directory containing real audio samples')
    parser.add_argument('--cloned-dir', type=str, default='data/cloned',
                       help='Directory containing cloned audio samples')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for classification (default: 0.5)')
    parser.add_argument('--distance-scale', type=float, default=10.0,
                       help='Distance normalization scale (default: 10.0)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Weights as "distance,threshold,statistical" (e.g., "0.2,0.5,0.3")')
    
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
    
    print("="*70)
    print("BATCH TEST - Testing All Files")
    print("="*70)
    
    batch_results = batch_test_all_files(
        real_dir=args.real_dir,
        cloned_dir=args.cloned_dir,
        threshold=args.threshold,
        weights=weights,
        distance_scale=args.distance_scale
    )
    
    print_summary(batch_results)
    
    return batch_results

if __name__ == "__main__":
    main()

