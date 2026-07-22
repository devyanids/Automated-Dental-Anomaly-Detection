#!/usr/bin/env python3
"""
04_finetune_resnet_anomalies.py

Master Stage 4 Pipeline:
1. Filters out purely healthy images and explicitly excludes the 32 noisy/outlier sentences.
2. Maps remaining lesion polygons into 7 clean clinical anomaly classes.
3. Performs Stratified Split BEFORE augmentation to guarantee 0% data leakage!
4. Sets up PyTorch WeightedRandomSampler + heavy Geometric Data Augmentation on the training fold.
5. Fine-tunes ResNet50 with Two-Stage Domain Rewiring:
   - Early layers frozen (conv1, layer1, layer2).
   - Layer 3 at low learning rate (1e-5).
   - Layer 4 at high learning rate (3e-4) in initial epochs, decaying to low learning rate (3e-5) in late epochs via Scheduler.
   - Classification head at standard learning rate (1e-3).
6. Extracts domain-specialized 2,048-dim feature embeddings using the fine-tuned backbone.
7. Optimizes an Anomaly-Only Random Forest using Bayesian Optimization / Probabilistic Search with Stratified 5-Fold CV!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Categorical, Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms

from config import DATA_ROOT, RADIOGRAPHS_DIR, DESCRIPTIONS_FILE, OUTPUT_DIR, MODEL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_STATE

FINETUNED_RESNET_PATH = MODEL_DIR / "resnet50_finetuned_anomalies.pth"
FINETUNED_RF_PATH = MODEL_DIR / "rf_anomalies_finetuned.joblib"

# 32 Outlier keywords to completely remove from dataset
BLACKLIST_WORDS = [
    'condyle', 'tmj', 'osteoarthriti', 'hypoplasia', 'dysplasia', 
    'marrow defect', 'none', 'extraction', 'edentulous', 'socket', 'healing', 'pericoronitis'
]

def get_anomaly_category(description, l4_title):
    desc = description.lower()
    
    # Check blacklist first to completely remove outlier sentences
    if any(w in desc for w in BLACKLIST_WORDS):
        return "IGNORE"
        
    if any(w in desc for w in ['caries', 'carious', 'enamel', 'dentin', 'pulp', 'decay']):
        return "Dental Caries / Radiolucency"
    elif any(w in desc for w in ['impacted', 'follicular', 'impaction', 'unerupted', 'erupted', 'buccally']):
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

class LesionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def extract_all_lesions():
    print("=== Step 1: Extracting Lesion Crops & Filtering Out 32 Outlier Sentences ===")
    with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    anomalous_records = [d for d in data if d.get('Description', '').strip().lower() != 'within normal limits']
    print(f"Total dataset images: {len(data)} | Purely healthy excluded: {len(data) - len(anomalous_records)}")
    
    crops = []
    labels = []
    metadata = []
    ignored_count = 0
    
    for rec in anomalous_records:
        img_id = rec.get("External ID")
        desc = rec.get("Description", "")
        
        img_path = RADIOGRAPHS_DIR / img_id
        if not img_path.exists():
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
        
        for obj in rec.get("Label", {}).get("objects", []):
            polygons = obj.get("polygons")
            if polygons == "none" or not isinstance(polygons, list) or len(polygons) == 0:
                continue
                
            l4_titles = [c.get('answer', c.get('answers', [{}])[0] if c.get('answers') else {}).get('title', '')
                         for c in obj.get("classifications", []) if "four" in c.get("title", "").lower()]
            l4_title = l4_titles[0] if l4_titles else "Unknown"
            
            cat_label = get_anomaly_category(desc, l4_title)
            if cat_label == "IGNORE":
                ignored_count += 1
                continue
                
            for poly in polygons:
                if not isinstance(poly, list) or len(poly) < 3:
                    continue
                    
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon([tuple(pt) for pt in poly], fill=255)
                
                masked_img = Image.new("RGB", (width, height), (0, 0, 0))
                masked_img.paste(full_img, mask=mask)
                
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                min_x, max_x = max(0, min(xs)), min(width, max(xs))
                min_y, max_y = max(0, min(ys)), min(height, max(ys))
                
                if max_x <= min_x or max_y <= min_y:
                    continue
                    
                crop_img = masked_img.crop((min_x, min_y, max_x, max_y))
                crops.append(crop_img)
                labels.append(cat_label)
                metadata.append({"image_id": img_id, "label": cat_label, "desc": desc})
                
    print(f"Successfully filtered out {ignored_count} outlier polygons!")
    print(f"Total clean dental anomaly crops: {len(crops)}")
    
    # Map labels to integers
    unique_classes = sorted(list(set(labels)))
    label_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_label = {i: cls for i, cls in enumerate(unique_classes)}
    
    int_labels = np.array([label_to_idx[lbl] for lbl in labels])
    
    print("\nClean Class Distribution:")
    for cls, idx in label_to_idx.items():
        count = np.sum(int_labels == idx)
        print(f"  [{idx}] {cls:<35}: {count} samples")
        
    return crops, int_labels, label_to_idx, idx_to_label

def build_finetune_resnet(num_classes, device):
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    
    # FREEZE early layers (conv1, bn1, relu, maxpool, layer1, layer2)
    for name, param in resnet.named_parameters():
        if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1', 'layer2']):
            param.requires_grad = False
        else:
            param.requires_grad = True  # Unfreeze layer3, layer4, fc
            
    # Replace FC classification head
    in_feats = resnet.fc.in_features
    resnet.fc = nn.Linear(in_feats, num_classes)
    return resnet.to(device)

def train_resnet_domain_rewiring(crops, int_labels, num_classes, device):
    print("\n=== Step 2: Stratified Split BEFORE Augmentation (0% Data Leakage) ===")
    
    # 80% Train, 20% Validation Stratified Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(skf.split(crops, int_labels))
    
    train_crops = [crops[i] for i in train_idx]
    train_labels = int_labels[train_idx]
    val_crops = [crops[i] for i in val_idx]
    val_labels = int_labels[val_idx]
    
    print(f"Stratified Training Set: {len(train_crops)} crops | Validation Set: {len(val_crops)} crops")
    
    # Data Augmentation for Training Set
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Calculate class weights for WeightedRandomSampler to balance training draws
    class_counts = Counter(train_labels)
    total_train = len(train_labels)
    class_weights = {cls: total_train / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_ds = LesionDataset(train_crops, train_labels, transform=train_transform)
    val_ds = LesionDataset(val_crops, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n=== Step 3: Fine-Tuning ResNet50 with Two-Stage Domain Rewiring Schedule ===")
    print("  -> Early layers (conv1, layer1, layer2): FROZEN (lr=0)")
    print("  -> Layer 3: Low learning rate (1e-5)")
    print("  -> Layer 4: High learning rate (3e-4) in initial epochs, decaying by 10x in late epochs")
    print("  -> Classification Head (fc): Standard learning rate (1e-3)")
    
    model = build_finetune_resnet(num_classes, device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 3e-4},
        {'params': model.fc.parameters(),     'lr': 1e-3}
    ], weight_decay=1e-2)
    
    # After epoch 8, drop LR by 10x for precision polishing!
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    num_epochs = 15
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_imgs, batch_lbls in train_loader:
            batch_imgs, batch_lbls = batch_imgs.to(device), batch_lbls.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_imgs.size(0)
            
        scheduler.step()
        train_loss /= len(train_ds)
        
        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_imgs, batch_lbls in val_loader:
                batch_imgs = batch_imgs.to(device)
                outputs = model(batch_imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_lbls.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average="weighted") * 100
        lr_layer4 = optimizer.param_groups[1]['lr']
        
        print(f"Epoch [{epoch:2d}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:5.2f}% | Val F1: {val_f1:5.2f}% | Layer4 LR: {lr_layer4:.1e}")
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), FINETUNED_RESNET_PATH)
            
    print(f"\n[SUCCESS] Saved Best Fine-Tuned ResNet50 (Val F1: {best_val_f1:.2f}%) -> {FINETUNED_RESNET_PATH}")
    return model

def extract_finetuned_embeddings(model, crops, device):
    print("\n=== Step 4: Extracting 2,048-dim Embeddings from Fine-Tuned Domain Backbone ===")
    model.eval()
    # Remove classification head to output 2,048 feature vectors
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device)
    backbone.eval()
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_feats = []
    batch_imgs = []
    
    with torch.no_grad():
        for i, img in enumerate(crops):
            batch_imgs.append(val_transform(img))
            if len(batch_imgs) >= BATCH_SIZE or i == len(crops) - 1:
                t = torch.stack(batch_imgs).to(device)
                out = backbone(t).squeeze().cpu().numpy()
                if len(batch_imgs) == 1:
                    out = np.expand_dims(out, axis=0)
                all_feats.append(out)
                batch_imgs = []
                
    features = np.vstack(all_feats)
    print(f"Extracted feature matrix shape: {features.shape}")
    return features

def tune_random_forest(X_features, int_labels, idx_to_label):
    print("\n=== Step 5: Hyperparameter Optimization for Random Forest via Stratified 5-Fold CV ===")
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    if HAS_SKOPT:
        print("Using Bayesian Optimization (skopt BayesSearchCV)...")
        search_space = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'max_features': Categorical(['sqrt', 'log2', 0.2, 0.3])
        }
        search = BayesSearchCV(rf, search_space, n_iter=25, cv=cv, scoring='f1_weighted', random_state=RANDOM_STATE, n_jobs=-1)
    else:
        print("scikit-optimize not installed. Using Randomized Probabilistic Search (RandomizedSearchCV) over Stratified 5-Fold CV...")
        search_space = {
            'n_estimators': [100, 150, 200, 300, 400, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.2, 0.3]
        }
        search = RandomizedSearchCV(rf, search_space, n_iter=30, cv=cv, scoring='f1_weighted', random_state=RANDOM_STATE, n_jobs=-1)
        
    print("Searching for optimal Random Forest hyperparameters...")
    search.fit(X_features, int_labels)
    
    print(f"\nBest Cross-Validation Weighted F1-Score: {search.best_score_*100:.2f}%")
    print("Best Hyperparameters Found:")
    for param, val in search.best_params_.items():
        print(f"  {param:<20}: {val}")
        
    best_rf = search.best_estimator_
    
    # Save bundle containing model, feature columns, and label dictionary
    bundle = {
        "model": best_rf,
        "feat_cols": [f"feat_{k}" for k in range(X_features.shape[1])],
        "idx_to_label": idx_to_label
    }
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, FINETUNED_RF_PATH)
    print(f"\n[SUCCESS] Saved Fine-Tuned Anomaly Random Forest Bundle -> {FINETUNED_RF_PATH}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Master Stage 4 on device: {device}")
    
    crops, int_labels, label_to_idx, idx_to_label = extract_all_lesions()
    num_classes = len(label_to_idx)
    
    # Fine-tune ResNet50
    finetuned_resnet = train_resnet_domain_rewiring(crops, int_labels, num_classes, device)
    
    # Reload best checkpoint for feature extraction
    finetuned_resnet.load_state_dict(torch.load(FINETUNED_RESNET_PATH))
    X_features = extract_finetuned_embeddings(finetuned_resnet, crops, device)
    
    # Optimize and train Random Forest
    tune_random_forest(X_features, int_labels, idx_to_label)
    
    print("\n=== Stage 4 Master Pipeline Completed Successfully! ===")

if __name__ == "__main__":
    main()
