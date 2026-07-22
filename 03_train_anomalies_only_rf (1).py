#!/usr/bin/env python3
"""
03_train_anomalies_only_rf.py

Refined Training Script:
1. Filters the Tufts Dental Database strictly for the 340 anomalous images (excluding the 660 purely healthy images).
2. For every anomaly polygon in expert.json, creates a binary mask to black out background noise outside the lesion.
3. Crops the masked lesion and extracts 2,048-dimensional visual feature embeddings using ResNet50.
4. Maps each polygon to a clean Diagnostic Anomaly Class using built-in hierarchical metadata and descriptions.
5. Trains an Anomaly-Only Random Forest classifier (0% normal bias, 100% anomaly specialization!).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from config import DATA_ROOT, RADIOGRAPHS_DIR, DESCRIPTIONS_FILE, OUTPUT_DIR, MODEL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_STATE

EXPERT_JSON = DESCRIPTIONS_FILE
ANOMALIES_MODEL_PATH = MODEL_DIR / "rf_anomalies_only.joblib"

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
    print("=== Refined Stage 3: Training Anomaly-Only Random Forest on Polygon Lesions ===")
    
    if not EXPERT_JSON.exists():
        raise FileNotFoundError(f"Cannot find expert.json at {EXPERT_JSON}")
        
    with open(EXPERT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Filter strictly for anomalous images
    anomalous_records = [d for d in data if d.get('Description', '').strip().lower() != 'within normal limits']
    print(f"Total dataset images: {len(data)}")
    print(f"Purely healthy images excluded: {len(data) - len(anomalous_records)}")
    print(f"Anomalous images to process: {len(anomalous_records)}")
    
    # Initialize ResNet50 feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ResNet50 on device: {device}...")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    X_features = []
    y_labels = []
    metadata = []
    
    print("Extracting polygon-masked visual features across abnormal radiographs...")
    batch_imgs = []
    batch_meta = []
    
    for rec in anomalous_records:
        img_id = rec.get("External ID")
        desc = rec.get("Description", "")
        
        # Locate image file
        img_path = RADIOGRAPHS_DIR / img_id
        if not img_path.exists():
            # Try lowercase/uppercase extension
            alt_path = RADIOGRAPHS_DIR / f"{Path(img_id).stem}.jpg"
            if alt_path.exists():
                img_path = alt_path
            else:
                continue
                
        try:
            full_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        width, height = full_img.size
        
        for obj_idx, obj in enumerate(rec.get("Label", {}).get("objects", [])):
            polygons = obj.get("polygons")
            if polygons == "none" or not isinstance(polygons, list) or len(polygons) == 0:
                continue
                
            # Extract hierarchical Level 4 metadata
            l4_titles = [c.get('answer', c.get('answers', [{}])[0] if c.get('answers') else {}).get('title', '')
                         for c in obj.get("classifications", []) if "four" in c.get("title", "").lower()]
            l4_title = l4_titles[0] if l4_titles else "Unknown"
            
            cat_label = get_anomaly_category(desc, l4_title)
            
            # Process each polygon outline in this object
            for poly in polygons:
                if not isinstance(poly, list) or len(poly) < 3:
                    continue
                    
                # Create polygon binary mask
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon([tuple(pt) for pt in poly], fill=255)
                
                # Apply mask to image (black out surrounding background bone/teeth)
                masked_img = Image.new("RGB", (width, height), (0, 0, 0))
                masked_img.paste(full_img, mask=mask)
                
                # Crop tight around polygon bounding box
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                min_x, max_x = max(0, min(xs)), min(width, max(xs))
                min_y, max_y = max(0, min(ys)), min(height, max(ys))
                
                if max_x <= min_x or max_y <= min_y:
                    continue
                    
                crop_img = masked_img.crop((min_x, min_y, max_x, max_y))
                
                batch_imgs.append(crop_img)
                batch_meta.append({"image_id": img_id, "label": cat_label, "l4": l4_title})
                
                if len(batch_imgs) >= BATCH_SIZE:
                    feats = extract_resnet_batch(resnet, batch_imgs, device, transform)
                    if len(batch_imgs) == 1:
                        feats = np.expand_dims(feats, axis=0)
                    for f, m in zip(feats, batch_meta):
                        X_features.append(f)
                        y_labels.append(m["label"])
                        metadata.append(m)
                    batch_imgs = []
                    batch_meta = []
                    
    # Process remaining batch
    if batch_imgs:
        feats = extract_resnet_batch(resnet, batch_imgs, device, transform)
        if len(batch_imgs) == 1:
            feats = np.expand_dims(feats, axis=0)
        for f, m in zip(feats, batch_meta):
            X_features.append(f)
            y_labels.append(m["label"])
            metadata.append(m)
            
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    
    print(f"Extracted {len(X_features)} polygon lesion embeddings.")
    print("\nClass Distribution in Anomaly-Only Training Set:")
    for cls_name, count in pd.Series(y_labels).value_counts().items():
        print(f"  {cls_name:<35}: {count}")
        
    # Split for internal validation
    X_train, X_val, y_train, y_val = train_test_split(X_features, y_labels, test_size=0.2, random_state=RANDOM_STATE, stratify=y_labels)
    
    print(f"\nTraining Anomaly-Only Random Forest on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    print(f"\nInternal Validation Accuracy: {accuracy_score(y_val, val_preds)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, zero_division=0))
    
    # Save trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ANOMALIES_MODEL_PATH)
    print(f"[SUCCESS] Saved Anomaly-Only Random Forest model -> {ANOMALIES_MODEL_PATH}")

if __name__ == "__main__":
    main()
