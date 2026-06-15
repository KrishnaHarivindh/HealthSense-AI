# HealthSense AI

![CI](https://github.com/KrishnaHarivindh/HealthSense-AI/actions/workflows/python-ci.yml/badge.svg)
![License](https://img.shields.io/github/license/KrishnaHarivindh/HealthSense-AI)
![Last Commit](https://img.shields.io/github/last-commit/KrishnaHarivindh/HealthSense-AI)
![Language](https://img.shields.io/github/languages/top/KrishnaHarivindh/HealthSense-AI)

AI-assisted chest X-ray risk analysis platform built for local CPU execution, clinical metadata fusion, explainable predictions, and interactive medical imaging review.

## Overview

HealthSense AI combines computer vision and structured-data machine learning to estimate pneumonia risk from chest X-ray images and optional patient metadata. The system is designed as a practical healthcare AI prototype with a Streamlit dashboard, ensemble scoring, Grad-CAM explainability, and downloadable prediction reports.

## Key Features

- Chest X-ray image upload and preprocessing
- CNN-based image classification pipeline
- Structured clinical metadata modeling with Random Forest and SVM
- Weighted ensemble layer for final disease-risk scoring
- Grad-CAM heatmaps for visual explainability
- Streamlit dashboard with prediction, metrics, explainability, and history tabs
- Local SQLite prediction history
- JSON and HTML report export
- CPU-friendly training configuration for local development

## Tech Stack

**Language:** Python  
**ML / AI:** TensorFlow, Keras, Scikit-Learn, XGBoost  
**Data:** Pandas, NumPy  
**Visualization:** Matplotlib, Grad-CAM  
**App:** Streamlit  
**Storage:** SQLite

## Project Structure

```text
HealthSense-AI/
  app/                  Streamlit application
  src/                  Training, preprocessing, evaluation, and ensemble logic
  healthsense/          Core reusable package modules
  data/                 Local data workspace, ignored from Git
  models/               Trained model artifacts, ignored from Git
  reports/              Generated reports and metrics, ignored from Git
  requirements.txt      Python dependencies
```

## Local Setup

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Train Models

```powershell
.\.venv\Scripts\python src\train.py --target-label Pneumonia --max-positive-samples 500 --epochs 3
```

## Run Dashboard

```powershell
.\.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

## Output Artifacts

- CNN model artifacts
- Tabular ML model artifacts
- Ensemble configuration
- Evaluation metrics
- Confusion matrix
- Prediction reports

## Notes

This repository is structured for portfolio review and local experimentation. Large datasets, trained models, logs, caches, and generated reports are excluded from version control.
