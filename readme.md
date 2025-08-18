# SiTrEx: Siamese Transformer for Exercise Feedback

This repository contains the official code for our paper:  
**SiTrEx: Siamese Transformer for Feedback and Posture Correction on Workout Exercises**

---

## 📌 Contents
- `sitrex/model.py` — Code for creating the Angle USefulness and Angle Similarity models.
- `sitrex/preprocessing.py` — Code for preprocessing and feature extraction.
- `train.ipynb` — Jupyter notebook for training SiTrEx on pose data.
- `inference.ipynb` — Jupyter notebook for running inference with pretrained weights.
- `requirements.txt` — List of Python dependencies.

---

## 🚀 Quick Start (Colab)

Click below to run the training notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asellam/sitrex/blob/main/train.ipynb)

---

## ⚙️ Installation (Local)

Clone the repository and install dependencies:

```bash
git clone https://github.com/asellam/sitrex.git
cd sitrex
pip install -r requirements.txt