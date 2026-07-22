#!/usr/bin/env python3
import json, joblib, torch, pandas as pd, numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from importlib import import_module
import torchvision.transforms as transforms

m = import_module("04_finetune_resnet_anomalies")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rf_bundle = joblib.load("outputs/models/rf_anomalies_finetuned.joblib")
svm_bundle = joblib.load("outputs/models/svm_anomalies_finetuned.joblib")
rf = rf_bundle["model"]
svm = svm_bundle["model"]
idx_to_label = rf_bundle["idx_to_label"]
feat_cols = rf_bundle["feat_cols"]

resnet = m.build_finetune_resnet(len(idx_to_label), device)
resnet.load_state_dict(torch.load("outputs/models/resnet50_finetuned_anomalies.pth", map_location=device, weights_only=True))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

annotations_json_path = Path("data/instances_Test_with_polygons.json")
test_images_dir = Path(r"C:\Users\JRK\Desktop\Devyani\Originals\Originals")
if not test_images_dir.exists():
    test_images_dir = Path(r"C:\Users\JRK\Desktop\Devyani\Originals")

with open(annotations_json_path, "r") as f:
    coco_data = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
img_id_to_file = {i["id"]: i["file_name"] for i in coco_data.get("images", [])}

# Load test descriptions mapping
gt_df = pd.read_csv("outputs/finetuned_blind_test_verification_report.csv") if Path("outputs/finetuned_blind_test_verification_report.csv").exists() else None
gt_desc_map = {}
if gt_df is not None:
    for _, r in gt_df.iterrows():
        gt_desc_map[f"{r['Image']}_{r['Box_ID']}"] = r["Ground_Truth_Description"]

ann_by_img = {}
for ann in coco_data.get("annotations", []):
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

rows = []
for img_id, anns in sorted(ann_by_img.items()):
    file_name = img_id_to_file.get(img_id, f"{img_id}.jpg")
    img_path = test_images_dir / file_name
    if not img_path.exists():
        continue
    full_img = Image.open(img_path).convert("RGB")
    width, height = full_img.size
    img_stem = Path(file_name).stem

    for ann in anns:
        box_id = cat_id_to_name.get(ann["category_id"], "Unknown")
        bbox = ann["bbox"]
        min_x, min_y, w, h = bbox
        max_x, max_y = min_x + w, min_y + h

        polygons = ann.get("segmentation", [])
        if isinstance(polygons, list) and len(polygons) > 0 and isinstance(polygons[0], list) and len(polygons[0]) >= 6:
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

        t = transform(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_vec = resnet(t).squeeze().cpu().numpy().reshape(1, -1)
        feat_df = pd.DataFrame(feat_vec, columns=feat_cols)

        rf_pred = idx_to_label[rf.predict(feat_df)[0]]
        svm_pred = idx_to_label[svm.predict(feat_df)[0]]
        desc = gt_desc_map.get(f"{img_stem}_{box_id}", "Clinical Finding")

        rows.append({
            "Image": img_stem,
            "Box_ID": box_id,
            "Doctor_Description": desc,
            "RF_Prediction": rf_pred,
            "SVM_Prediction": svm_pred
        })

df = pd.DataFrame(rows)
df.to_csv("outputs/rf_vs_svm_test_comparison.csv", index=False)

print("\n=== EXTERNAL BLIND TEST DATASET (OS01 - OS06 | 21 Lesion Polygons): RF vs. SVM ===\n")
print(f"{'Image':<6} | {'Box':<8} | {'Doctor Description':<38} | {'Random Forest':<33} | {'SVM Classifier':<33}")
print("-" * 125)
for _, r in df.iterrows():
    print(f"{r['Image']:<6} | {r['Box_ID']:<8} | {r['Doctor_Description'][:37]:<38} | {r['RF_Prediction']:<33} | {r['SVM_Prediction']:<33}")
