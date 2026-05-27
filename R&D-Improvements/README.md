# Urban Sound Classification — Road Traffic & Acoustic Events

End-to-end pipeline for classifying urban acoustic events from real-world recordings.  
The system distinguishes 10 classes of road traffic sounds and acoustic events using a lightweight 2D CNN trained on log-mel spectrograms.

---

## Authors

BAJJA Ilyas | BENEDETTI Giovanni | LAFKIH Rihab | LEKOUARA Ania

---

## Classes

| Index | Label        | Description                       |
|-------|--------------|-----------------------------------|
| 0     | BG           | Background / silence              |
| 1     | MC           | Motorcycle                        |
| 2     | Car          | Passenger car                     |
| 3     | Bus          | Bus                               |
| 4     | Truck        | Heavy truck                       |
| 5     | Tram         | Tram                              |
| 6     | Horn         | Horn / car honking                |
| 7     | Dog          | Dog barking                       |
| 8     | Construction | Construction noise                |
| 9     | Siren        | Emergency siren                   |

---

## Project structure

```
project/
│
├── data/
│   ├── IDMT_Traffic/
│   │   ├── annotation/
│   │   │   └── idmt_traffic_all.txt
│   │   └── audio/
│   │       └── *.wav
│   ├── MELAUDIS_BG/
│   │   └── *.wav                       # long continuous background recordings
│   ├── MELAUDIS_Vehicles/
│   │   └── {location}/
│   │       └── *.wav                   # isolated vehicle events, label in filename
│   ├── us8k_filtered/
│   │   ├── train/
│   │   │   └── {Construction,Dog,Horn,Siren}/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── segments/                       # produced by notebook 02
│   │   └── {BG,Car,MC,...}/
│   │       └── *.wav
│   ├── features/                       # produced by notebook 03
│   │   ├── logmel/
│   │   │   └── {class}/*.npy
│   │   └── pcen/
│   │       └── {class}/*.npy
│   ├── features_manifest.csv           # produced by notebook 04
│   ├── segments_manifest.csv           # produced by notebook 02
│   ├── norm_stats_logmel.npz           # produced by notebook 04
│   └── norm_stats_pcen.npz
│
├── audio/                              # field recordings for notebook 06
│   └── *.wav
├── Enregistrements_05_05_2026_.../     # field recordings session 1
│   └── *.wav
├── Enregistrements_30_04/              # field recordings session 2
│   └── *.wav
│
├── checkpoints/                        # produced by notebook 05
│   └── logmel_best-v1.ckpt
├── logs/                               # TensorBoard / CSV training logs
├── results_eval/                       # produced by notebook 06
│
├── 01_data_exploration.ipynb
├── 02_segmentation_pipeline.ipynb
├── 03_feature_extraction.ipynb
├── 04_dataset_dataloader.ipynb
├── 05_model_training.ipynb
├── 06_evaluation_real_recordings.ipynb
│
└── README.md
```

---

## Notebooks

The notebooks are designed to be run in order. Each one produces files that the next one depends on.

### 01 — Data exploration

Explores the three source databases without modifying any audio.  
Parses labels from IDMT-Traffic (annotation file + filename convention), Melaudis Vehicles (filename regex), and UrbanSound8K filtered.  
Checks sample rates, channel counts, and durations across all sources. Produces a coverage table showing how many clips are available per class per database, and plots class distributions.

### 02 — Segmentation pipeline

Converts all raw audio files into a unified set of 5-second mono WAV clips at 22050 Hz.  
All clips are center-padded or trimmed to the target length: the event is placed at the centre of the window, with silence on either side. Long Melaudis background recordings are sliced with a sliding window.  
Each class is capped at 1000 clips (randomly sampled, fixed seed). Writes `data/segments/` and `data/segments_manifest.csv`.

### 03 — Feature extraction

Converts every 5-second WAV segment into a pre-computed spectrogram saved as a `.npy` file.  
Computes two feature types: **log-mel** (64 mel bins, shape `64x216`) and **PCEN** (Per-Channel Energy Normalization), which is more robust to varying recording levels and stationary background noise.  
Pre-computation is mandatory on CPU — on-the-fly extraction during training would be 10-20x slower.  
Writes `data/features/{logmel,pcen}/{class}/*.npy`.

### 04 — Dataset, DataLoader & train/val/test split

Builds the PyTorch `Dataset` and PyTorch Lightning `DataModule`.  
The split is done at the **file level** (not clip level) to prevent leakage: two clips from the same original recording cannot end up in different splits. Stratification is by both label and source database to ensure every split contains clips from all three sources.  
Split ratios: 70% train / 15% val / 15% test.  
Computes normalisation statistics (mean and std) on the training set only and saves them to `data/norm_stats_{feature_type}.npz`. Writes the split column into `data/features_manifest.csv`.

### 05 — Model architecture & training

Defines and trains a small 2D CNN on the pre-computed spectrograms.  
The architecture treats each spectrogram `(1, 64, 216)` as a grayscale image through four convolutional blocks followed by a two-layer classifier head (~2M parameters).  
Training uses inverse-frequency class weights, Adam optimiser, ReduceLROnPlateau scheduler, and early stopping on validation loss. SpecAugment (time and frequency masking) is applied during training.  
Saves the best checkpoint to `checkpoints/`.

### 06 — Fine-tuning & evaluation on field recordings

Adapts the trained model to field recordings collected with a real microphone.  
The backbone is frozen; only the classifier head is re-trained on the field data, augmented (pitch shift, time stretch, additive noise) for under-represented classes.  
Inference uses a sliding window (2s, step 0.5s) with confidence-weighted voting: windows with low maximum probability contribute less to the final decision.  
Evaluates accuracy and balanced accuracy per class, plots confusion matrices and probability distributions.

---

## Installation

Python 3.10+ is required.

```bash
pip install torch torchvision torchaudio pytorch-lightning
pip install librosa soundfile numpy pandas matplotlib seaborn scikit-learn tqdm
```

For GPU training, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the CUDA-enabled build matching your driver version.

To monitor training with TensorBoard:

```bash
pip install tensorboard
tensorboard --logdir logs/
```

---

## Data sources

- **IDMT-Traffic** — Fraunhofer IDMT, annotated road traffic recordings (car, motorcycle, bus, truck, background)
- **Melaudis** — Annotated urban vehicle recordings across multiple locations
- **UrbanSound8K** (filtered subset) — Urban acoustic events: construction, dog, horn, siren
