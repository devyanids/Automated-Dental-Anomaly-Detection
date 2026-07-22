#!/usr/bin/env python3
"""
05_predict_finetuned_test_images.py

Stage 5 (Fine-Tuned Pipeline):
1. Loads our domain-specialized ResNet50 backbone (fine-tuned with Two-Stage Domain Rewiring on 572 clean dental lesions).
2. Loads our Bayesian-Optimized Anomaly Random Forest (70.68% Stratified CV F1-score).
3. Evaluates blindly on the 21 test bounding boxes across OS01.jpg to OS06.jpg.
4. Generates side-by-side verification table (Ground Truth vs. Fine-Tuned Model Prediction).
5. Draws custom annotated comparison bounding boxes directly onto your test radiographs!
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import joblib

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from config import MODEL_DIR, OUTPUT_DIR, IMG_SIZE
from src.text_lookup import build_text_lookup_table

def build_finetuned_backbone(model_path, device):
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet.fc = nn.Linear(resnet.fc.in_features, 7)  # 7 clean anomaly classes
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    backbone.eval()
    return backbone

def extract_finetuned_feature(backbone, img_crop, device, transform):
    t = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        out = backbone(t).squeeze().cpu().numpy()
    if out.ndim == 0:
        out = np.expand_dims(out, axis=0)
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 5: Blind Inference with Fine-Tuned Domain Specialist (Device: {device}) ===")
    
    test_images_dir = Path(r"C:\Users\JRK\Desktop\Devyani\Originals\Originals")
    if not test_images_dir.exists():
        test_images_dir = Path(r"C:\Users\JRK\Desktop\Devyani\Originals")
        
    annotations_json_path = Path("data/instances_Test_with_polygons.json")
    if not annotations_json_path.exists():
        annotations_json_path = Path(r"C:\Users\JRK\Desktop\Devyani\Annotations\annotations\instances_Test.json")
    test_descriptions_path = Path("data/test_descriptions.txt")
    
    resnet_path = MODEL_DIR / "resnet50_finetuned_anomalies.pth"
    rf_path = MODEL_DIR / "rf_anomalies_finetuned.joblib"
    
    if not resnet_path.exists() or not rf_path.exists():
        print("[Error] Fine-tuned models not found! Please run 04_finetune_resnet_anomalies.py first.")
        return
        
    print("Loading Ground-Truth Text Lookup Table for evaluation check...")
    gt_text_df = build_text_lookup_table(test_descriptions_path)
    gt_map = {f"{r['image_id']}_{r['finding_id']}": r['finding_phrase'] for _, r in gt_text_df.iterrows()}
    
    print(f"Loading Fine-Tuned Random Forest from {rf_path}...")
    bundle = joblib.load(rf_path)
    clf = bundle["model"]
    feat_cols = bundle["feat_cols"]
    idx_to_label = bundle.get("idx_to_label", {})
    
    print(f"Loading Domain-Specialized ResNet50 Backbone from {resnet_path}...")
    backbone = build_finetuned_backbone(resnet_path, device)
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with open(annotations_json_path, "r") as f:
        coco_data = json.load(f)
        
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    img_id_to_file = {i["id"]: i["file_name"] for i in coco_data.get("images", [])}
    
    out_dir = OUTPUT_DIR / "annotated_test_images_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ann_by_img = {}
    for ann in coco_data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
        
    results = []
    
    print("\nStarting blind inference across 21 test bounding boxes...")
    print("=" * 115)
    print(f"{'Image':<8} | {'Box ID':<8} | {'Ground Truth (Text Description)':<40} | {'Fine-Tuned Model Prediction':<38}")
    print("=" * 115)
    
    for img_id, anns in sorted(ann_by_img.items()):
        file_name = img_id_to_file.get(img_id, f"{img_id}.jpg")
        img_path = test_images_dir / file_name
        
        if not img_path.exists():
            print(f"[Warning] Test image {file_name} not found at {img_path}")
            continue
            
        try:
            full_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Could not load image {file_name}: {e}")
            continue
            
        draw = ImageDraw.Draw(full_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        for ann in anns:
            box_id = cat_id_to_name.get(ann["category_id"], "Unknown")
            bbox = ann["bbox"]  # [x, y, w, h]
            min_x, min_y, w, h = bbox
            max_x, max_y = min_x + w, min_y + h
            
            # Check if polygon segmentation masks exist!
            polygons = ann.get("segmentation", [])
            if isinstance(polygons, list) and len(polygons) > 0 and isinstance(polygons[0], list) and len(polygons[0]) >= 6:
                width, height = full_img.size
                mask = Image.new("L", (width, height), 0)
                mask_draw = ImageDraw.Draw(mask)
                for poly in polygons:
                    if isinstance(poly, list) and len(poly) >= 6:
                        mask_draw.polygon([tuple(poly[i:i+2]) for i in range(0, len(poly), 2)], fill=255)
                masked_img = Image.new("RGB", (width, height), (0, 0, 0))
                masked_img.paste(full_img, mask=mask)
                crop_img = masked_img.crop((min_x, min_y, max_x, max_y))
            else:
                crop_img = full_img.crop((min_x, min_y, max_x, max_y))
            
            # Extract fine-tuned domain features
            feat_vec = extract_finetuned_feature(backbone, crop_img, device, val_transform)
            feat_df = pd.DataFrame([feat_vec], columns=feat_cols)
            
            pred_idx = clf.predict(feat_df)[0]
            if isinstance(pred_idx, (int, np.integer)) and pred_idx in idx_to_label:
                pred_label = idx_to_label[pred_idx]
            else:
                pred_label = str(pred_idx)
                
            img_stem = Path(file_name).stem
            gt_key = box_id
            gt_desc = gt_map.get(gt_key, "Unknown / Unmapped")
            
            print(f"{img_stem:<8} | {box_id:<8} | {gt_desc[:38]:<40} | {pred_label:<38}")
            
            results.append({
                "Image": img_stem,
                "Box_ID": box_id,
                "Ground_Truth_Description": gt_desc,
                "FineTuned_Prediction": pred_label
            })
            
            # Draw bounding box and label
            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=4)
            tag_text = f"{box_id}: {pred_label}"
            
            if hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), tag_text, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            else:
                tw, th = draw.textsize(tag_text, font=font)
                
            draw.rectangle([min_x, max(0, min_y - th - 6), min_x + tw + 8, min_y], fill="red")
            draw.text((min_x + 4, max(0, min_y - th - 4)), tag_text, fill="white", font=font)
            
        out_img_path = out_dir / f"{Path(file_name).stem}_finetuned_annotated.png"
        full_img.save(out_img_path)
        
    print("=" * 115)
    
    df_res = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "finetuned_blind_test_verification_report.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved complete 21-box verification report to: {csv_path}")
    print(f"[SUCCESS] Saved annotated radiograph inspection images to: {out_dir}")

if __name__ == "__main__":
    main()
