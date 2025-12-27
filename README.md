# Cloned Voice Recognition

## Project Overview
This project is designed for analyzing and recognizing cloned (deepfake) and real voice samples. It provides tools for data preparation, audio processing, and waveform analysis to help distinguish between real and cloned audio.

## Codebase Structure

```
├── analysis.ipynb           # Jupyter notebook for waveform analysis and visualization
├── clone_real_data.py       # Script to prepare/clone real data for experiments
├── clone.ipynb              # Notebook for cloning experiments
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
├── my/                      # User-specific experiments/data
│   └── cloning/
│       ├── enes/
│       │   └── cloned/      # Cloned samples for user 'enes'
│       └── ibo/
│           ├── cloned/      # Cloned samples for user 'ibo'
│           └── orijinal/    # Original samples for user 'ibo'
```

## Data Structure

- **data/real/**: Contains real audio samples, organized by speaker. Each speaker folder may include a `meta.json` file with metadata.
- **data/cloned/**: Contains cloned (deepfake) audio samples, organized similarly by speaker.
- **data/manifests/**: Intended for manifest or metadata files describing datasets.
- **my/cloning/**: User-specific experiments, with subfolders for each user and their cloned/original samples.

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