# Ad Click Prediction Service

## Project Overview
This repository features an end-to-end MLOps project designed to predict user advertisement clicks. It focuses on reproducible training, controlled releases, and CI/CD quality gates, adhering to **MLOps Level 2** standards.

## Problem Definition
The project addresses a binary classification task to estimate the click-through probability (CTR) using user demographics and session data.
- **Task:** Supervised Learning (Binary Classification)
- **Target:** Clicked on Ad (0/1)
- **Goal:** Build an operational pipeline for high-accuracy predictions.

## MLOps Scope (Implemented)
As the MLOps Engineer, I have implemented the following technical foundations:
* **Experiment Tracking:** Integrated MLflow to log all hyperparameters and metrics (Accuracy: 0.95, F1: 0.949).
* **Model Governance:** Established an MLflow Model Registry workflow, promoting models from "None" to "Production" (v1-v5).
* **Headless CI/CD Support:** Refactored the training pipeline to support non-GUI environments by replacing `plt.show()` with automated artifact logging via `plt.savefig()`.
* **Fault Tolerance:** Implemented XGBoost checkpointing (`xgb_checkpoint.json`) to ensure training resilience.
* **Repository Hygiene:** Decoupled local experiment logs (`mlruns`) and datasets from version control to maintain a lean production codebase.

## Development Setup

1. **Environment Initialization:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
