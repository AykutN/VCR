# Cloned Voice Recognition

## Project Overview
This project is designed for analyzing and recognizing cloned (deepfake) and real voice samples. It provides tools for data preparation, audio processing, and waveform analysis to help distinguish between real and cloned audio.

## Codebase Structure

```
├── analysis.ipynb           # Jupyter notebook for waveform analysis and visualization
├── clone_real_data.py       # Script to prepare/clone real data for experiments
├── convert_audio.ipynb      # Notebook for audio format conversion
├── record_sentences.py      # Script to record sentences for dataset
├── requirements.txt         # Python dependencies
├── data/                    # Main data directory
│   ├── README.md            # Data directory documentation
│   ├── cloned/              # Cloned (deepfake) audio samples
│   │   ├── sample1_ar/      # Cloned samples for speaker 'sample1_ar'
│   │   └── sample1_tr/      # Cloned samples for speaker 'sample1_tr'
│   ├── manifests/           # Metadata or manifest files
│   └── real/                # Real audio samples
│       ├── sample1_ar/      # Real samples for speaker 'sample1_ar'
│       │   └── meta.json    # Metadata for 'sample1_ar'
│       └── sample1_tr/      # Real samples for speaker 'sample1_tr'
│           └── meta.json    # Metadata for 'sample1_tr'

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

1. Prepare your data in the `data/real` and `data/cloned` directories, following the structure above.
2. Use the provided notebooks and scripts for analysis, conversion, and recording.
3. Run `analysis.ipynb` to visualize and compare real vs. cloned audio waveforms.

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
- Fourier / spectral features: spectral centroid, spectral rolloff, zero-crossing rate, spectral bandwidth.
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
- Features used: MFCC (13), Delta, Delta-Delta, spectral centroid, rolloff, zero-crossing rate, spectral bandwidth; each summarized by mean/std/skew/kurtosis (~200+ features total).

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

### Notes / Next Steps

- The threshold can be tuned (e.g., 0.36–0.37) if you want higher real accuracy at the expense of cloned recall.
- Consider adding further features or improving feature normalization for better robustness.
