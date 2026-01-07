# Cloned Voice Recognition

## Project Overview
This project is designed for analyzing and recognizing cloned (deepfake) and real voice samples. It provides tools for data preparation, audio processing, and waveform analysis to help distinguish between real and cloned audio.

## Codebase Structure

```
├── analysis.ipynb           # Jupyter notebook for waveform analysis and visualization
├── app.py                  # Flask web application for deepfake detection
├── batch_test.py           # Batch testing script for rule-based detection
├── train_ml_models.py      # ML model training script (Logistic Regression & SVM)
├── ml_detector.py          # ML-based detection module
├── hybrid_detector.py      # Hybrid detection (rule-based + ML)
├── export_predictions.py  # Export predictions in CSV format for competition
├── clone_real_data.py      # Script to prepare/clone real data for experiments
├── record_sentences.py     # Script to record sentences for dataset
├── analyze_scores.py       # Score distribution analysis
├── optimize_simple.py      # Parameter optimization script
├── requirements.txt        # Python dependencies
├── templates/               # HTML templates for web interface
│   └── index.html
├── static/                  # Static files (CSS, JS)
│   ├── css/
│   └── js/
├── models/                  # Trained ML models (created after training)
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── scaler.pkl
├── data/                    # Main data directory
│   ├── cloned/              # Cloned (deepfake) audio samples
│   │   └── [speaker_name]/  # Cloned samples organized by speaker
│   └── real/                # Real audio samples
│       └── [speaker_name]/  # Real samples organized by speaker
│           └── meta.json    # Metadata for speaker

```

## Data Structure

- **data/real/**: Contains real audio samples, organized by speaker. Each speaker folder may include a `meta.json` file with metadata.
- **data/cloned/**: Contains cloned (deepfake) audio samples, organized similarly by speaker.
- **data/manifests/**: Intended for manifest or metadata files describing datasets.

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

- `librosa`: Audio processing and feature extraction
- `matplotlib`: Plotting and visualization
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning models (Logistic Regression, SVM)
- `flask`: Web framework for the interface
- `jupyter`: For running notebooks

To install all dependencies, first create and activate a virtual environment:

```powershell
# On Windows (PowerShell)
.\.venv310\Scripts\Activate.ps1
```
or
```bash
# On Unix/macOS
source ./.venv310/Scripts/activate
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

1. Prepare your data in the `data/real` and `data/cloned` directories, following the structure above.
2. Use `record_sentences.py` to record real voice samples.
3. Use `clone_real_data.py` to generate cloned versions of the recordings.

### 2. ML Model Training

**Important:** Before using ML-based detection, you need to train the models using your dataset.

```bash
python train_ml_models.py --real-dir data/real --cloned-dir data/cloned --output models/
```

This script will:
- Extract features from all real and cloned audio files
- Train Logistic Regression and SVM classifiers
- Save trained models to the `models/` directory
- Display training results and accuracy metrics

**Parameters:**
- `--real-dir`: Directory containing real audio samples (default: `data/real`)
- `--cloned-dir`: Directory containing cloned audio samples (default: `data/cloned`)
- `--output`: Output directory for saved models (default: `models`)
- `--test-size`: Proportion of test set (default: 0.2)

**Note:** Make sure you have both real and cloned audio files in the specified directories before training.

### 3. Detection Methods

The system supports three detection methods:

1. **Rule-Based Detection** (no training required):
   ```bash
   python batch_test.py --threshold 0.34
   ```

2. **ML-Based Detection** (requires trained models):
   ```python
   from ml_detector import detect_with_ml
   result = detect_with_ml('path/to/audio.wav', models_dir='models')
   ```

3. **Hybrid Detection** (combines rule-based + ML):
   ```python
   from hybrid_detector import detect_hybrid
   result = detect_hybrid('path/to/audio.wav', real_dir='data/real', models_dir='models')
   ```

### 4. Web Interface

Start the web application:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

The web interface allows you to:
- Upload audio files (WAV, MP3, FLAC, OGG, M4A)
- Choose detection method (Hybrid, Rule-Based, or ML-Based)
- View detailed detection results with scores and confidence

### 5. Analysis and Visualization

Run `analysis.ipynb` to visualize and compare real vs. cloned audio waveforms.

## Notes

- Ensure your audio files are in a supported format (e.g., WAV).
- Update paths in scripts/notebooks as needed for your data organization.

---
For more details, see the comments in each script or notebook.

---

## System Explanation (Detailed)

This section explains the detection approach and system design used in this repository.

### Contents
- Overview
- Key concepts and features
- How the system works
- What we changed and results
- Settings and parameters

### Overview

This project detects cloned (deepfake) voice samples using rule-based comparisons rather than a learned ML classifier.

Core idea:
- Extract features from real (reference) audio
- Extract same features from test audio
- Compare features and score differences
- Large differences → likely cloned; small differences → likely real

### Key Concepts

- MFCC (Mel-Frequency Cepstral Coefficients): 13 coefficients representing perceptual spectral shape.
- Delta and Delta-Delta: first and second derivatives of MFCCs to capture dynamics.
- Fourier / spectral features: spectral centroid, spectral rolloff, zero-crossing rate, spectral bandwidth, RMS energy, spectral flatness.
- Statistical summaries: mean, std, skewness, kurtosis computed per feature.
- Distance metric: Euclidean distance between feature vectors.
- Threshold: decision cutoff applied to the combined score.

### How the System Works

1. Data preparation: record real voices and generate cloned versions (see `record_sentences.py` and `clone_real_data.py`).
2. Feature extraction per file: MFCCs, deltas, spectral features, and stats.
3. Build a reference distribution from real samples.
4. For each test file, compute feature distances and threshold deviations vs. the reference.
5. Combine distance, threshold-exceed counts, and statistical measures into a hybrid score (range 0–1).
6. Decide: score ≥ threshold → Fake, else → Real.

### What We Did / Results

- Initial threshold (0.5) produced poor cloned detection (only ~5% detected).
- After analyzing score distributions we selected an optimal threshold of **0.34**.
- With threshold 0.34: Real accuracy ≈ 85% (17/20), Cloned accuracy = 100% (20/20), Overall ≈ 92.5%.

### Settings and Parameters

- Threshold: default **0.34** (changeable via `batch_test.py --threshold`).
- Hybrid weights: (distance, threshold, statistical) default `(0.3, 0.4, 0.3)`.
- Distance scale: default `10.0` (normalization factor).
- MFCC parameters: `n_mfcc=13`, `hop_length=512`, `n_fft=2048`.
- Features used: 
  - MFCC (13 coefficients), Delta, Delta-Delta
  - Spectral features: spectral centroid, spectral rolloff, zero-crossing rate (ZCR), spectral bandwidth
  - Energy features: RMS energy
  - Spectral shape: spectral flatness
  - Each feature summarized by mean/std/skewness/kurtosis (~200+ features total)

### Usage Examples

Single file test (Python):
```python
from batch_test import detect_deepfake

result = detect_deepfake('path/to/audio.wav', real_dir='data/real', threshold=0.34)
print('Is Fake:', result['is_fake'])
print('Score: {:.4f}'.format(result['score']))
```

Batch test (CLI):
```bash
python batch_test.py --threshold 0.34
```

Analyze/optimize parameters:
```bash
python analyze_scores.py
python optimize_simple.py
python quick_optimize.py
```

Export predictions for competition (CSV format):
```bash
python export_predictions.py test_directory/ --output predictions.csv --threshold 0.34
```

### Competition Format

For competition submissions, use the `export_predictions.py` script to generate predictions in the required CSV format:

```bash
python export_predictions.py test_directory/ --output predictions.csv --threshold 0.34
```

The output format is:
```csv
filename,prediction,confidence
file_001.wav,0,0.95
file_002.wav,1,0.82
```

Where:
- `prediction`: 0 = Real, 1 = Fake
- `confidence`: 0.0–1.0 confidence score

### Evaluation Metrics

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of predicted fakes that are actually fake
- **Recall (Fake Recall)**: Proportion of actual fakes correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Proportion of real samples incorrectly classified as fake
- **Confusion Matrix**: Detailed breakdown of predictions

These metrics are automatically computed and displayed when running `batch_test.py`.

### Notes / Next Steps

- The threshold can be tuned (e.g., 0.36–0.37) if you want higher real accuracy at the expense of cloned recall.
- All required features from the project specification are now included (MFCC, Delta, Delta-Delta, Spectral features, RMS Energy, Spectral Flatness).
- The system is ready for competition submission with CSV export functionality.
