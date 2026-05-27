# Urban Sound Classifier

AI-based system for **acoustic classification of road traffic sounds**, built as the project S8.

---

## Project Overview

Road traffic noise is a major environmental and public health issue.
This project aims to build a **Python-based AI prototype** capable of:

* Classifying road noise sources from audio recordings
* Working with real-world data in urban environments

---

## Approach

The pipeline follows these main steps:

1. **Data Collection**  
   Audio recordings collected using Raspberry Pi and microphone.

2. **Preprocessing & Segmentation**  
   Audio cleaning and event-based segmentation.

3. **Feature Extraction**  
   Extraction of audio features such as MFCCs and spectrograms.

4. **Modeling**  
   Machine Learning and Deep Learning models for sound classification.

5. **Evaluation**  
   Validation of the model on real-world traffic sound data.

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
* and others

---

## Repository Structure

The repository is organized into three main folders:

```text
.
├── data/                     # Datasets used for training and testing
├── current-model/             # Current and presented solution
│   └── Vehicle_Audio_Classifier_V12.ipynb
├── R&D-Improvements/          # Alternative experimental version with a different approach
├── requirements.txt           # Python dependencies
└── README.md
```

### Folders Description

* **data/**  
  Contains the datasets used in the project, such as IDMT Traffic, Melaudis, and UrbanSound8K.  
  The datasets are not included in the repository due to size constraints.

* **current-model/**  
  Contains the current version of the solution presented for the project.  
  This is the main implementation of the Urban Sound Classifier.

* **R&D-Improvements/**  
  Contains an alternative research and development version of the project, based on a different approach that achieved very promising results.

* **requirements.txt**  
  Contains the Python libraries required to run the project.

---

## Previous Version

This repository replaces an earlier version of the project.

Previous repository:  
[Project_S8](https://github.com/gioviben/Project_S8)

---