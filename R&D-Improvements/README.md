# Urban Sound Classification: Road Traffic and Acoustic Events

End-to-end pipeline for classifying urban acoustic events from real-world recordings.
The system distinguishes **10 classes** of road traffic sounds and acoustic events using a lightweight 2D CNN trained on log-mel spectrograms.

---

## Authors

BAJJA Ilyas · BENEDETTI Giovanni · LAFKIH Rihab · LEKOUARA Ania

---

## Table of Contents

1. [Classes](#classes)
2. [Project structure](#project-structure)
3. [Notebooks overview](#notebooks-overview)
4. [Installation](#installation)
5. [Data sources & required files](#data-sources--required-files)
6. [Quick start](#quick-start)
7. [Expected outputs per notebook](#expected-outputs-per-notebook)
8. [Reusing the trained checkpoint](#reusing-the-trained-checkpoint)
9. [Configuration reference](#configuration-reference)

---

## Classes

| Index | Label        | Description                  |
|-------|--------------|------------------------------|
| 0     | BG           | Background / silence         |
| 1     | MC           | Motorcycle                   |
| 2     | Car          | Passenger car                |
| 3     | Bus          | Bus                          |
| 4     | Truck        | Heavy truck                  |
| 5     | Tram         | Tram                         |
| 6     | Horn         | Horn / car honking           |
| 7     | Dog          | Dog barking                  |
| 8     | Construction | Construction noise           |
| 9     | Siren        | Emergency siren              |

---

## Project structure

```
project/
│
├── data/
│   ├── IDMT_Traffic/                       ← required external dataset (see Data sources)
│   │   ├── annotation/
│   │   │   └── idmt_traffic_all.txt
│   │   └── audio/
│   │       └── *.wav
│   ├── MELAUDIS_BG/                        ← required external dataset
│   │   └── *.wav
│   ├── MELAUDIS_Vehicles/                  ← required external dataset
│   │   └── {location}/
│   │       └── *.wav
│   ├── us8k_filtered/                      ← required external dataset
│   │   ├── train/{Construction,Dog,Horn,Siren}/*.wav
│   │   ├── val/{Construction,Dog,Horn,Siren}/*.wav
│   │   └── test/{Construction,Dog,Horn,Siren}/*.wav
│   │
│   ├── segments/                           ← produced by notebook 02
│   │   └── {BG,Car,MC,...}/
│   │       └── *.wav
│   ├── features/                           ← produced by notebook 03
│   │   ├── logmel/{class}/*.npy
│   │   └── pcen/{class}/*.npy
│   ├── segments_manifest.csv               ← produced by notebook 02
│   ├── features_manifest.csv               ← produced by notebooks 03 & 04
│   ├── norm_stats_logmel.npz               ← produced by notebook 04
│   └── norm_stats_pcen.npz                 ← produced by notebook 04
│
├── audio/                                  ← field recordings for notebook 06
│   └── *.wav
├── audios_16h_00h/                         ← field recording session (Evening)
│   └── *.wav
├── audios_00h_08h/                         ← field recording session (Night)
│   └── *.wav
├── audios_08h_16h/                         ← field recording session (Day)
│   └── *.wav
│
├── checkpoints/                            ← produced by notebook 05
│   └── logmel_best-v1.ckpt
├── logs/                                   ← TensorBoard / CSV training logs
├── results_eval/                           ← produced by notebook 06
│
├── 01_data_exploration.ipynb
├── 02_segmentation_pipeline.ipynb
├── 03_feature_extraction.ipynb
├── 04_dataset_dataloader.ipynb
├── 05_model_training.ipynb
├── 06_24h_acoustic_analysis.ipynb
│
├── README.md
└── user_guide.pdf
```

---

## Notebooks overview

The notebooks are designed to be run **in order**. Each one produces files that the next one depends on.

### 01 - Data Exploration

Explores the three source databases without modifying any audio.
Parses labels from IDMT-Traffic (annotation file + filename convention), Melaudis Vehicles (filename regex), and UrbanSound8K filtered.
Checks sample rates, channel counts, and durations across all sources. Produces a coverage table showing how many clips are available per class per database, and plots class distributions and per-class waveform/spectrogram examples.

**Input:** raw `data/` directory with the three source databases  
**Output:** no files written (exploration only)

---

### 02 - Segmentation Pipeline

Converts all raw audio files into a unified set of 5-second mono WAV clips at 22 050 Hz.
All clips are center-padded or trimmed to the target length. Long Melaudis background recordings are sliced with a sliding window.
Each class is capped at **1 000 clips** (randomly sampled, fixed seed = 42).

**Input:** raw source databases  
**Output:** `data/segments/{class}/*.wav`, `data/segments_manifest.csv`  
**Expected runtime:** 5–15 min on CPU (I/O bound)

---

### 03 - Feature Extraction

Converts every 5-second WAV segment into a pre-computed spectrogram saved as a `.npy` file.
Two feature types are computed in parallel:
- **log-mel**: 64 mel bins, shape `(64, 216)`, output in dB
- **PCEN**: Per-Channel Energy Normalization, more robust to level variation and stationary background noise

Pre-computing features is mandatory: on-the-fly extraction during training would be 10–20× slower on CPU.

**Input:** `data/segments_manifest.csv`, `data/segments/`  
**Output:** `data/features/{logmel,pcen}/{class}/*.npy`, `data/features_manifest.csv` (partial; the `split` column is added in NB 04)  
**Expected runtime:** 10–20 min on CPU

---

### 04 - Dataset, DataLoader and Train/Val/Test Split

Builds the PyTorch `Dataset` and PyTorch Lightning `DataModule`.

**Split strategy:** file-level stratified split to prevent data leakage: two clips from the same original recording will never appear in different splits. Stratification is by both label and source database.

| Split | Ratio |
|-------|-------|
| Train | 70 %  |
| Val   | 15 %  |
| Test  | 15 %  |

Normalisation statistics (mean and std) are computed on the **training set only** via Welford's online algorithm (constant memory, no need to load all spectrograms at once).

**Input:** `data/features_manifest.csv`, `data/features/`  
**Output:** `data/norm_stats_{logmel,pcen}.npz`, `data/features_manifest.csv` updated with `split` column

---

### 05 - Model Architecture and Training

Defines and trains a small 2D CNN on the pre-computed spectrograms.

**Architecture:** four `ConvBlock` layers (Conv2d → BatchNorm → ReLU → MaxPool) followed by a two-layer fully connected classifier head. Input shape: `(1, 64, 216)`. Approximately **2 M parameters**.

**Training setup:**
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Optimiser: Adam
- Scheduler: ReduceLROnPlateau (monitor `val_loss`)
- Regularisation: SpecAugment (time masking + frequency masking) during training
- Early stopping on `val_loss` (patience configurable)
- Checkpoint: best validation loss saved to `checkpoints/`

**Input:** `data/features_manifest.csv`, `data/norm_stats_logmel.npz`  
**Output:** `checkpoints/logmel_best-v1.ckpt`, `logs/`  
**Expected runtime:** ~1–2 min per epoch on CPU; early stopping typically at epoch 20–30

---

### 06 - 24-Hour Acoustic Analysis

Applies the trained model to continuous field recordings collected over a full 24-hour period, divided into three sessions:

| Folder           | Period  | Hours        |
|------------------|---------|--------------|
| `audios_16h_00h` | Evening | 16:00 → 00:00 |
| `audios_00h_08h` | Night   | 00:00 → 08:00 |
| `audios_08h_16h` | Day     | 08:00 → 16:00 |

Inference uses a **sliding window** (2 s, step 0.5 s) with confidence-weighted voting. Produces 14 figures covering:
- Sound level (dBFS) over 24 h (P10/P90 envelope + annotated peaks)
- Vehicle activity per hour + dBFS secondary axis
- Evening / Night / Day comparison
- Raw and normalised class heatmaps
- Global event distribution (pie + bar)
- Multi-metric temporal evolution
- Noisiest vehicle statistics (violin plots + boxplots)
- Precise Gantt timeline (second resolution)
- dBFS × traffic correlation
- Radar profile per period
- 100% stacked area chart
- CNN confidence over time
- Interactive audio player with spectrogram

**Input:** `checkpoints/logmel_best-v1.ckpt`, `audios_16h_00h/`, `audios_00h_08h/`, `audios_08h_16h/`  
**Output:** figures inline in notebook, `results_eval/` (optional export)

---

## Installation

Python **3.10+** is required.

```bash
# Core dependencies
pip install torch torchvision torchaudio pytorch-lightning

# Audio & signal processing
pip install librosa soundfile

# Data & visualisation
pip install numpy pandas matplotlib seaborn scikit-learn tqdm

# Notebook utilities
pip install ipywidgets
```

For GPU training, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the CUDA-enabled build matching your driver version.

To monitor training with TensorBoard:

```bash
pip install tensorboard
tensorboard --logdir logs/
```

> **Tested versions:** Python 3.10, PyTorch 2.1, pytorch-lightning 2.1, librosa 0.10, numpy 1.26

---

## Data sources & required files

You need to obtain and place the following datasets manually before running notebook 01.

| Dataset | Classes provided | Where to get it |
|---------|-----------------|-----------------|
| **IDMT-Traffic** | Car, MC, Bus, Truck, BG | [Fraunhofer IDMT](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html) |
| **Melaudis** | Car, MC, Bus, Truck, Tram, BG | Contact the Melaudis project team |
| **UrbanSound8K** (filtered subset) | Construction, Dog, Horn, Siren | [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html): filter to the 4 listed classes and place in `us8k_filtered/{train,val,test}/{class}/` |

The UrbanSound8K filtered directory must already be split into `train/val/test` before running. If starting from the raw UrbanSound8K archive, re-run the original fold-based split and keep only the four classes listed above.

---

## Quick start

### Option A: Full pipeline from scratch

```bash
# 1. Place source databases in data/ (see Data sources above)
# 2. Run notebooks in order:
jupyter notebook 01_data_exploration.ipynb        # ~7 min
jupyter notebook 02_segmentation_pipeline.ipynb   # ~4 min
jupyter notebook 03_feature_extraction.ipynb      # ~7 min
jupyter notebook 04_dataset_dataloader.ipynb      # ~4 min
jupyter notebook 05_model_training.ipynb          # ~1h 20 min (CPU)
jupyter notebook 06_24h_acoustic_analysis.ipynb   # ~5 h
```

### Option B: Skip to training (segments and features already computed)

```bash
# Requires: data/features_manifest.csv, data/features/, data/norm_stats_logmel.npz
jupyter notebook 05_model_training.ipynb
```

### Option C: Skip to analysis (checkpoint already available)

```bash
# Requires: checkpoints/logmel_best-v1.ckpt, audios_16h_00h/, audios_00h_08h/, audios_08h_16h/
jupyter notebook 06_24h_acoustic_analysis.ipynb
```

---

## Expected outputs per notebook

| Notebook | Files written | Approx. size |
|----------|--------------|-------------|
| 01 | none | none |
| 02 | `data/segments/` + `segments_manifest.csv` | ~2–4 GB |
| 03 | `data/features/` + `features_manifest.csv` | ~500 MB |
| 04 | `norm_stats_logmel.npz`, `norm_stats_pcen.npz`, updated `features_manifest.csv` | < 1 MB |
| 05 | `checkpoints/logmel_best-v1.ckpt`, `logs/` | ~50 MB |
| 06 | Figures inline + optional `results_eval/` exports | none |

---

## Reusing the trained checkpoint

To run inference or evaluation without re-training, load the checkpoint directly in notebook 06 or in a standalone script:

```python
from pathlib import Path
import torch
from your_module import AudioClassifier  # class defined in NB 05

model = AudioClassifier.load_from_checkpoint(
    "checkpoints/logmel_best-v1.ckpt",
    n_classes=10,
    class_weights=torch.ones(10),   # weights not used at inference
)
model.eval()
```

The checkpoint includes the full model state. You do **not** need to re-run notebooks 01–04 to use it.

---

## Configuration reference

Key parameters are defined at the top of each notebook under `## 0. Imports & configuration`. The most commonly adjusted values are listed below.

| Parameter | Default | Notebook | Description |
|-----------|---------|----------|-------------|
| `TARGET_SR` | 22 050 | 02 | Output sample rate (Hz) |
| `TARGET_DUR` | 5.0 | 02 | Clip duration (seconds) |
| `CAP` | 1 000 | 02 | Max clips per class |
| `SEED` | 42 | 02, 04 | Random seed for reproducibility |
| `N_MELS` | 64 | 03 | Number of mel filterbank bins |
| `N_FFT` | 2 048 | 03 | FFT window size |
| `HOP_LENGTH` | 512 | 03 | STFT hop length |
| `FEATURE_TYPE` | `logmel` | 04, 05 | `logmel` or `pcen` |
| `BATCH_SIZE` | 64 | 04, 05 | DataLoader batch size |
| `LR` | 1e-3 | 05 | Initial learning rate |
| `MAX_EPOCHS` | 60 | 05 | Training epoch ceiling |
| `PATIENCE` | 10 | 05 | Early stopping patience |

---

## Data sources

- **IDMT-Traffic**: Fraunhofer IDMT, annotated road traffic recordings (Car, MC, Bus, Truck, BG)
- **Melaudis**: Annotated urban vehicle recordings across multiple locations (Car, MC, Bus, Truck, Tram, BG)
- **UrbanSound8K** (filtered subset): Urban acoustic events: Construction, Dog, Horn, Siren
