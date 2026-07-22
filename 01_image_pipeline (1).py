# %% [markdown]
# # Stage 1 — Image pipeline (object detection → ResNet50 transfer learning → embeddings)
#
# Run this file top-to-bottom in VS Code with the Python + Jupyter extensions
# (each `# %%` block is a runnable cell — "Run Cell" / Shift+Enter).
#
# Output: `outputs/image_features.csv`, one row per detected tooth crop.
# This is the ONLY artifact the later text/fusion stages depend on, so you
# can iterate on everything below without ever touching text_lookup.py.

# %%
import sys
sys.path.append("..")  # if running this file from inside a subfolder, adjust as needed

from pathlib import Path
import pandas as pd
from PIL import Image

from config import RADIOGRAPHS_DIR, SEGMENTATION_DIR, OUTPUT_DIR, IMG_SIZE, BATCH_SIZE, IMAGE_FEATURES_CSV
from src.tooth_detector import MaskBasedDetector, FasterRCNNDetector, get_boxes_for_image, crop_and_save
from src.resnet_features import build_model, fine_tune, extract_features

# %% [markdown]
# ## 1. Object detection — locate tooth ROIs on every radiograph

# %%
crops_dir = OUTPUT_DIR / "tooth_crops"
mask_detector = MaskBasedDetector(SEGMENTATION_DIR / "teeth_mask")
fallback_detector = FasterRCNNDetector(score_threshold=0.5)

image_paths = sorted(Path(RADIOGRAPHS_DIR).glob("*.*"))
# image_paths = sorted(Path(RADIOGRAPHS_DIR).glob("*.*"))[:5]  # test batch
print(f"Found {len(image_paths)} radiographs in {RADIOGRAPHS_DIR}")

manifest_rows = []
for img_path in image_paths:
    image_id = img_path.stem
    image = Image.open(img_path).convert("RGB")
    boxes = get_boxes_for_image(image_id, image, mask_detector, fallback_detector)
    saved_paths = crop_and_save(image, boxes, crops_dir)
    for p in saved_paths:
        manifest_rows.append({"filepath": str(p), "image_id": image_id})

crops_manifest = pd.DataFrame(manifest_rows)
print(f"Detected {len(crops_manifest)} tooth crops across {crops_manifest['image_id'].nunique()} images")
crops_manifest.to_csv(OUTPUT_DIR / "crops_manifest.csv", index=False)
crops_manifest.head()

# %% [markdown]
# ## 2. (Optional) transfer learning — fine-tune ResNet50 on labelled crops
#
# Skip this cell and go straight to feature extraction with ImageNet weights
# if you don't have per-crop labels yet. Once you do (e.g. derived from the
# `diagnosis` column later, or a manual subset), point `labelled_df` at a
# dataframe with `filepath`, `image_id`, `label` (integer class id) columns.

# %%
USE_FINE_TUNING = False  # flip to True once you have labelled_df ready

if USE_FINE_TUNING:
    from sklearn.model_selection import train_test_split

    labelled_df = crops_manifest.copy()
    labelled_df["label"] = 0  # <-- replace with your real integer labels

    train_df, val_df = train_test_split(labelled_df, test_size=0.2, random_state=42)
    num_classes = labelled_df["label"].nunique()

    model = fine_tune(train_df, val_df, num_classes=num_classes,
                       img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_epochs=10)
else:
    model = build_model(num_classes=2, freeze_backbone=True)  # ImageNet-pretrained backbone only

# %% [markdown]
# ## 3. Extract embeddings for every crop and save the feature table

# %%
features_df = extract_features(model, crops_manifest, IMG_SIZE, BATCH_SIZE)
features_df.to_csv(IMAGE_FEATURES_CSV, index=False)
print(f"Wrote {len(features_df)} embeddings ({features_df.shape[1]-1} dims each) -> {IMAGE_FEATURES_CSV}")
features_df.head()

# %%
