# 🛡️ Explainable Machine Learning for Spam and Phishing Detection

> A three-class classifier (ham / spam / phishing) for SMS and email messages, built with classical machine learning and LIME-based explanations.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/license-academic-lightgrey)

---

## 📖 About

This is an MSc dissertation project at **Trinity College Dublin**, School of Computer Science and Statistics. It explores whether classical machine learning models — combined with explainability techniques — can effectively detect spam and phishing messages across both SMS and email channels.

| | |
|---|---|
| **Author** | Pratik Pramod Mattha |
| **Supervisor** | Dr. Van-Dinh Nguyen |
| **Programme** | MSc Computer Science — Future Networked Systems |
| **Institution** | Trinity College Dublin |

---

## 🎯 Goals

- Build a 3-class classifier: **Ham**, **Spam**, **Phishing**
- Train classical models: **Naïve Bayes**, **Logistic Regression**, **SVM**
- Handle class imbalance with **SMOTE** and class weights
- Add **LIME** explanations for every prediction
- Optional **Streamlit** web demo

---

## 📚 Datasets

| Dataset | Source | Type |
|---|---|---|
| SMS Spam Collection | UCI ML Repository | SMS — ham, spam |
| SpamAssassin Public Corpus | Apache | Email — ham, spam |
| SMS Phishing Dataset | Mishra & Soni (Mendeley Data, 2022) | SMS — ham, spam, smishing |
| Nazario Phishing Corpus | monkey.org | Email — phishing |

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/pratikmattha/spam-phishing-detection.git
cd spam-phishing-detection

# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
spam-phishing-detection/
│
├── config.py           # Project paths and settings
├── requirements.txt    # Python dependencies
├── .gitignore
│
├── data/               # Datasets
│   ├── raw/            # Original downloads (never modified)
│   ├── interim/        # Intermediate processing
│   └── processed/      # Final cleaned, train/test ready
│
├── src/                # Source code
│   ├── data/           # Data loading and cleaning
│   ├── features/       # Feature engineering
│   ├── models/         # Model training
│   ├── evaluation/     # Metrics and confusion matrices
│   └── explainability/ # LIME implementation
│
├── models/             # Trained model artefacts
├── results/            # Figures, metrics, reports
├── app/                # Streamlit demo (optional)
├── docs/               # Proposal, interim report
└── tests/              # Unit tests
```

---

## 🚧 Status

Work in progress. Current phase: **data collection and preprocessing**.

See `docs/` for the proposal and interim report.

---

## 📄 License

Academic use only. Datasets retain their original licenses — see `docs/` for citations.

---

<p align="center">Made for the MSc dissertation at Trinity College Dublin 🎓</p>