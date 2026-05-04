# Urban Sound Classifier
AI-based system for **acoustic classification of road traffic sounds**, built as the project S8.

---

## Project Overview

Road traffic noise is a major environmental and public health issue.
This project aims to build a **Python-based AI prototype** capable of:

* Classifying road noise sources from audio recordings
* Working with real-world data (urban environment)

---

## Approach

The pipeline follows these main steps:

1. **Data Collection** (Other team)

   * Audio recordings (Raspberry Pi + microphone) 

2. **Preprocessing & Segmentation**

   * Audio cleaning
   * Event-based segmentation

3. **Feature Extraction**

   * MFCCs
   * Spectrograms

4. **Modeling**

   * Machine Learning / Deep Learning models (CNN/LSTM)

5. **Evaluation**

   * Validation on real-world data

---

## Dataset

The model is trained using a combination of datasets:

* IDMT Traffic
* Melaudis
* UrbanSound8K

Classes include:

* Car
* Bike
* Horn
* Siren
* Dog
* Construction
* (and others)

---

## Repository Structure (basic)

```
.
├── Vehicle_Audio_Classifier_V12.ipynb   # Main notebook (latest solution)
├── data/                                # Datasets (not included)
├── models/                              # Saved models (optional)
└── README.md
```

---

## Previous Version

This repository replaces an earlier version of the project.

👉 Previous repository:
[Project_S8](https://github.com/gioviben/Project_S8)

---