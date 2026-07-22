#!/usr/bin/env python3
"""
04_train_svm_anomalies.py
=========================
Standalone script to train a Support Vector Machine (SVM) classifier on our
fine-tuned ResNet50 domain embeddings.

Why a separate script?
- Keeps 04_finetune_resnet_anomalies.py (Random Forest) 100% untouched so you can
  compare RF vs. SVM side-by-side or switch back anytime.
- SVMs rely on hyperplane margin distances (||w*x - b||), so this script includes
  a StandardScaler pipeline to normalize the 2,048-dim embeddings before SVM fit.
- Performs RandomizedSearchCV hyperparameter optimization across 5-Fold Stratified CV
  and prints the full classification metrics table across all 7 anomaly classes.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

import torch

from config import OUTPUT_DIR, MODEL_DIR, RANDOM_STATE
from importlib import import_module

FINETUNED_RESNET_PATH = MODEL_DIR / "resnet50_finetuned_anomalies.pth"
FINETUNED_SVM_PATH = MODEL_DIR / "svm_anomalies_finetuned.joblib"

def tune_and_evaluate_svm(X_features, int_labels, idx_to_label):
    print("\n=== Optimizing & Evaluating SVM Classifier on Fine-Tuned ResNet50 Features ===")
    
    # 1. Build pipeline with StandardScaler + Support Vector Classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    
    # 2. Define SVM Hyperparameter Search Space
    search_space = {
        'svm__C': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'svm__kernel': ['rbf', 'linear', 'poly'],
        'svm__gamma': ['scale', 'auto', 0.001, 0.0001],
        'svm__degree': [2, 3]  # Only used when kernel='poly'
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    search = RandomizedSearchCV(
        pipeline,
        search_space,
        n_iter=25,
        cv=cv,
        scoring='f1_weighted',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    print("Searching for optimal SVM hyperparameters across 5-Fold Stratified CV...")
    search.fit(X_features, int_labels)
    
    best_pipeline = search.best_estimator_
    print(f"\nBest Cross-Validation Weighted F1-Score: {search.best_score_*100:.2f}%")
    print("Best Hyperparameters Found:")
    for param, val in search.best_params_.items():
        print(f"  {param:<20}: {val}")
        
    # 3. Compute out-of-fold cross-validated predictions using the best pipeline
    print("\nComputing 5-Fold Stratified Cross-Validation Predictions across all 572 Lesions...")
    cv_preds = cross_val_predict(best_pipeline, X_features, int_labels, cv=cv, n_jobs=-1)
    
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    print("\n=== SVM VALIDATION DATASET METRICS (5-Fold Stratified Cross-Validation) ===\n")
    print(classification_report(int_labels, cv_preds, target_names=target_names, digits=4))
    
    acc = accuracy_score(int_labels, cv_preds)
    print(f"Overall 5-Fold CV SVM Accuracy: {acc*100:.2f}%")
    
    # 4. Save best SVM bundle
    bundle = {
        "model": best_pipeline,
        "feat_cols": [f"feat_{k}" for k in range(X_features.shape[1])],
        "idx_to_label": idx_to_label
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, FINETUNED_SVM_PATH)
    print(f"\n[SUCCESS] Saved Fine-Tuned SVM Bundle -> {FINETUNED_SVM_PATH}")
    return best_pipeline

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Standalone Stage 4 SVM Training on device: {device}")
    
    # Import extraction utilities from 04_finetune_resnet_anomalies
    m = import_module("04_finetune_resnet_anomalies")
    crops, int_labels, label_to_idx, idx_to_label = m.extract_all_lesions()
    num_classes = len(label_to_idx)
    
    # Load Fine-Tuned ResNet50
    print(f"Loading fine-tuned ResNet50 backbone from {FINETUNED_RESNET_PATH}...")
    finetuned_resnet = m.build_finetune_resnet(num_classes, device)
    finetuned_resnet.load_state_dict(torch.load(FINETUNED_RESNET_PATH, map_location=device))
    
    # Extract 2,048-dim embeddings
    X_features = m.extract_finetuned_embeddings(finetuned_resnet, crops, device)
    
    # Tune and Evaluate SVM
    tune_and_evaluate_svm(X_features, int_labels, idx_to_label)
    
    print("\n=== Stage 4 Standalone SVM Training Completed Successfully! ===")

if __name__ == "__main__":
    main()
