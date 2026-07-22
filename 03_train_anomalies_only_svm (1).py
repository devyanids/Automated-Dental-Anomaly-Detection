#!/usr/bin/env python3
"""
03_train_anomalies_only_svm.py
==============================
Standalone script to train a Support Vector Machine (SVM) classifier on
out-of-the-box (Stage 3) ResNet50 visual feature embeddings.

Leaves 03_train_anomalies_only_rf.py untouched!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from config import DATA_ROOT, RADIOGRAPHS_DIR, DESCRIPTIONS_FILE, OUTPUT_DIR, MODEL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_STATE

EXPERT_JSON = DESCRIPTIONS_FILE
ANOMALIES_SVM_PATH = MODEL_DIR / "svm_anomalies_only.joblib"

def get_anomaly_category(description, l4_title):
    desc = description.lower()
    if any(w in desc for w in ['caries', 'carious', 'enamel', 'dentin', 'pulp', 'decay']):
        return "Dental Caries / Radiolucency"
    elif any(w in desc for w in ['impacted', 'follicular', 'impaction', 'ununerupted', 'erupted', 'buccally']):
        return "Impacted / Widened Follicle"
    elif any(w in desc for w in ['root fragment', 'root stump', 'stump', 'remnant']):
        return "Remnant Root Fragment"
    elif any(w in desc for w in ['bone island', 'osteosclerosis', 'radiopacity', 'sclerotic', 'condensing', 'dense']):
        return "Dense Bone Island / Radiopacity"
    elif any(w in desc for w in ['cyst', 'pseudocyst', 'osteoma']) or l4_title == 'Benign Cyst Neoplasia':
        return "Benign / Cyst Lesion"
    elif any(w in desc for w in ['trauma', 'fracture', 'osteopenia', 'osteoporosis']) or l4_title in ['Trauma', 'Metabolic/Systemic']:
        return "Trauma / Systemic Condition"
    else:
        return "Periapical Radiolucency / Osteitis"

def extract_resnet_batch(model, img_list, device, transform):
    model.eval()
    tensors = [transform(img) for img in img_list]
    batch_t = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model(batch_t)
    return features.squeeze().cpu().numpy()

def main():
    print("=== Stage 3 Standalone SVM Training on Out-of-the-Box ResNet50 Features ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(EXPERT_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()

    X_list, y_list = [], []

    for entry in raw_data:
        img_name = entry.get("External ID", "")
        img_path = RADIOGRAPHS_DIR / img_name
        if not img_path.exists():
            continue

        label_data = entry.get("Label", {})
        objects = label_data.get("objects", [])

        try:
            full_img = Image.open(img_path).convert("RGB")
            w_orig, h_orig = full_img.size
        except Exception:
            continue

        for obj in objects:
            poly = obj.get("polygon", [])
            if len(poly) < 3:
                continue

            desc = obj.get("value", "")
            l4_title = ""
            for cls_info in obj.get("classifications", []):
                if isinstance(cls_info, dict) and "four" in cls_info.get("title", "").lower():
                    ans = cls_info.get("answer", {})
                    if isinstance(ans, dict):
                        l4_title = ans.get("title", "")

            cat = get_anomaly_category(desc, l4_title)

            pts = []
            for pt in poly:
                pts.append((pt.get("x", 0) * w_orig, pt.get("y", 0) * h_orig))

            mask = Image.new("L", (w_orig, h_orig), 0)
            ImageDraw.Draw(mask).polygon(pts, fill=255)
            masked_img = Image.composite(full_img, Image.new("RGB", (w_orig, h_orig), (0, 0, 0)), mask)

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = max(0, int(min(xs))), min(w_orig, int(max(xs)))
            min_y, max_y = max(0, int(min(ys))), min(h_orig, int(max(ys)))

            if (max_x - min_x) < 10 or (max_y - min_y) < 10:
                continue

            crop = masked_img.crop((min_x, min_y, max_x, max_y))
            X_list.append(crop)
            y_list.append(cat)

    labels = pd.Series(y_list)
    valid_mask = labels.map(labels.value_counts()) >= 5
    X_list = [X_list[i] for i, v in enumerate(valid_mask) if v]
    y_list = [y_list[i] for i, v in enumerate(valid_mask) if v]

    feats = []
    for i in range(0, len(X_list), BATCH_SIZE):
        batch = X_list[i:i+BATCH_SIZE]
        f = extract_resnet_batch(resnet, batch, device, transform)
        if len(batch) == 1:
            f = np.expand_dims(f, axis=0)
        feats.append(f)
    X_features = np.vstack(feats)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])

    search_space = {
        'svm__C': [0.1, 1.0, 5.0, 10.0, 50.0],
        'svm__kernel': ['rbf', 'linear', 'poly'],
        'svm__gamma': ['scale', 'auto', 0.001]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(pipeline, search_space, n_iter=15, cv=cv, scoring='f1_weighted', random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_features, y_list)

    best_pipeline = search.best_estimator_
    print(f"\nStage 3 Best SVM Cross-Validation Weighted F1: {search.best_score_*100:.2f}%")

    cv_preds = cross_val_predict(best_pipeline, X_features, y_list, cv=cv, n_jobs=-1)
    print("\n=== Stage 3 Out-of-the-Box ResNet50 + SVM Validation Report ===\n")
    print(classification_report(y_list, cv_preds, digits=4))

    bundle = {
        "model": best_pipeline,
        "feat_cols": [f"feat_{k}" for k in range(X_features.shape[1])],
        "idx_to_label": sorted(list(set(y_list)))
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ANOMALIES_SVM_PATH)
    print(f"[SUCCESS] Saved Stage 3 SVM -> {ANOMALIES_SVM_PATH}")

if __name__ == "__main__":
    main()
