# SiTrEx: Siamese Transformer for Exercise Feedback

This repository contains the official code for our paper:  
**SiTrEx: Siamese Transformer for Feedback and Posture Correction on Workout Exercises**

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.neucom.2025.131703.svg)](https://doi.org/10.1016/j.neucom.2025.131703)


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

Click below to run the inference notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asellam/sitrex/blob/main/inference.ipynb)

---

## ⚙️ Installation (Local)

Clone the repository and install dependencies:

```bash
git clone https://github.com/asellam/sitrex.git
cd sitrex
pip install -r requirements.txt

## 📖 Citation

If you use **SiTrEx** in your research, please cite:

> Sellam, A., Kassimi, D., Djebana, A., & Mokhtari, S.  
> *SiTrEx: Siamese transformer for feedback and posture correction on workout exercises.*  
> Neurocomputing, Volume 658, 2025, 131703.  
> https://doi.org/10.1016/j.neucom.2025.131703

### BibTeX

```bibtex
@article{SELLAM2025131703,
  title     = {SiTrEx: Siamese transformer for feedback and posture correction on workout exercises},
  journal   = {Neurocomputing},
  volume    = {658},
  pages     = {131703},
  year      = {2025},
  issn      = {0925-2312},
  doi       = {https://doi.org/10.1016/j.neucom.2025.131703},
  url       = {https://www.sciencedirect.com/science/article/pii/S0925231225023756},
  author    = {Abdellah Sellam and Dounya Kassimi and Abdelhadi Djebana and Sara Mokhtari},
  keywords  = {Workout exercise classification, Exercise posture correction, Transformer, Siamese neural network, One-shot learning},
  abstract  = {Applying Machine Learning and Deep Learning techniques to sequences of Human Pose Landmarks to recognize workout exercises and count repetitions is widely studied in the computer vision literature. However, existing approaches suffer from two major problems. The first issue is that they lack the ability to provide detailed feedback on the postures performed by the athletes or provide feedback for a limited range of exercises using hand-designed rules and algorithms. The second problem is that these approaches consider only a predefined set of exercises and do not generalize to exercises outside their training data, which limits their usability. In this paper, we aim to address these two shortcomings by proposing a one-shot learning approach that utilizes Siamese Transformers to provide detailed feedback on individual human joints and can generalize to new exercises that are not present in the used dataset. The proposed configuration of the Siamese Transformer model deviates from its standard use in that it outputs a vector of similarity indicators rather than a single similarity score. Additionally, an accompanying binary classification Transformer model is used to assess the usefulness of different parts of the human pose for the input exercise without prior knowledge of the exercise itself. These properties allow the proposed approach to be used in general-purpose fitness applications and coach/athlete training platforms. The proposed approach achieved a 5-fold cross-validation test accuracy of 94.4%±0.8 on the collected dataset.}
}