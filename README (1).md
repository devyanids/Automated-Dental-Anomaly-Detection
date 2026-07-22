# Dental Anomaly Classification — Image + Text Fusion Pipeline

Multimodal pipeline for the **Tufts Dental Database (TDD)**:
object detection → ResNet50 transfer learning (image branch) fused with a
keyword lookup table built from expert text descriptions (text branch),
classified with a Random Forest.

Dataset: https://www.kaggle.com/datasets/tommyngx/the-tufts-dental-database-2022
(1000 panoramic radiographs with tooth/abnormality segmentation masks and
free-text expert findings — see Panetta et al., IEEE JBHI 2021, for the
official schema).

## Why this structure won't force you to rewrite the image code

The three stages only talk to each other through two flat CSV files keyed
by `image_id`:

```
01_image_pipeline.py  ──▶  outputs/image_features.csv   (image_id, feat_0 … feat_2047)
02_text_lookup.py      ──▶  outputs/text_lookup_table.csv (image_id, keywords, diagnosis, …)
03_fusion_train.py      merges the two CSVs on image_id → trains RandomForest
```

`src/tooth_detector.py` and `src/resnet_features.py` (the image branch)
never import anything from `src/text_lookup.py`, and vice versa. The only
file that imports both is `src/fusion_model.py`. That means:

- You can build, debug, and re-run the entire image branch today.
- Weeks later, adding the text branch is just: write `text_lookup_table.csv`,
  then run `03_fusion_train.py`. Nothing in `01_image_pipeline.py` changes.
- If you swap ResNet50 for another backbone, or the keyword regex for an
  NLP model, only that one file changes — the CSV contract stays the same.

## Setup (VS Code + Python + Jupyter extensions)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

1. Download & unzip the Kaggle dataset, then edit the 4 path constants at
   the top of `config.py` to match your folder layout.
2. Open `01_image_pipeline.py` in VS Code, select your `.venv` interpreter,
   and run the `# %%` cells one at a time (Jupyter extension gives you an
   interactive window with plots/dataframes inline).
3. When you're ready to add text, do the same with `02_text_lookup.py`.
4. Run `03_fusion_train.py` to merge both and train the classifier.

## File map

```
config.py                 # <- edit dataset paths here, nothing else
src/
  tooth_detector.py        # object detection: mask-derived boxes, with a
                            # pretrained Faster R-CNN fallback where masks
                            # are missing
  resnet_features.py       # ResNet50: build_model / fine_tune / extract_features
  text_lookup.py            # regex parser: raw description lines ->
                            # image_id, tooth_number, diagnosis, keywords
  fusion_model.py            # merges both CSVs, trains RandomForestClassifier,
                            # prints classification report + feature importances
01_image_pipeline.py      # Stage 1 notebook (image only)
02_text_lookup.py         # Stage 2 notebook (text only)
03_fusion_train.py        # Stage 3 notebook (fusion + RF)
outputs/                  # all generated CSVs, crops, and the saved model
```

## Notes on the object detector

TDD includes expert-annotated tooth segmentation masks
(`Segmentation/teeth_mask/`). `MaskBasedDetector` converts each mask into
tight bounding boxes per labelled region — this is more reliable ground
truth than training your own detector from scratch, and is what
`tooth_detector.py` uses by default. A pretrained torchvision Faster R-CNN
(`FasterRCNNDetector`) is included as an automatic fallback for any image
missing a mask, and can also be swapped in entirely (or fine-tuned) if your
assignment specifically requires demonstrating a trained detector rather
than mask-derived boxes.

## Notes on the text keyword extraction

`src/text_lookup.py` parses lines like:

```
S01_a_Impacted 38: Impaction with 38
S02_a_Ill defined radiolucency seen involving the mesial root of 16: Chronic rarefying osteitis with 16
```

into `image_id`, `finding_id`, `tooth_number`, `finding_phrase`,
`diagnosis`, and a multi-hot `keywords` column matched against a small
clinical vocabulary (impaction, caries, radiolucency, osteitis, edentulous,
etc. — extend `KEYWORD_VOCAB` freely). Point `DESCRIPTIONS_FILE` in
`config.py` at whatever file your Kaggle download actually names the
per-image findings (`.json`, `.csv`, or `.xlsx` are all handled).

## Kaggle reference notebooks

The dataset's own Kaggle "Code" tab (linked above) has several community
notebooks doing tooth segmentation and abnormality classification on TDD —
worth skimming for how others handled the mask/label file formats
specifically, since Kaggle's dataset page renders via JavaScript and
notebook titles/links can't be scraped from outside Kaggle. Open the Code
tab directly in your browser and sort by "Most Votes" to find the most
battle-tested preprocessing code for this dataset.
