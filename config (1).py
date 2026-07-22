"""
config.py
=========
Single place to point the whole pipeline at your local copy of the
Tufts Dental Database (TDD). Nothing else in the project should hard-code
a path -- everything imports from here.

Expected folder layout after you download the Kaggle dataset
(tommyngx/the-tufts-dental-database-2022) and unzip it:

    DATA_ROOT/
        Radiographs/            # panoramic X-ray images, e.g. 1.JPG ... 1000.JPG
        Segmentation/
            teeth_mask/          # tooth segmentation masks (optional, used by tooth_detector)
            maxillomandibular/   # jaw ROI masks (optional)
        Expert/
            expert.json  or  descriptions.csv   # free-text abnormality descriptions
                                                  # (the "S01_a_Impacted 38: ..." lines
                                                  #  you pasted come from this file)

If your extracted folder names differ, only edit the constants below --
you will not need to touch src/text_lookup.py or src/tooth_detector.py.
"""

from pathlib import Path

# ---- EDIT THESE FOUR LINES FOR YOUR MACHINE ---------------------------------
# DATA_ROOT = Path(r"./dental_pipeline/data/the-tufts-dental-database-2022")     # unzip location
# RADIOGRAPHS_DIR = DATA_ROOT / "Radiographs"/"Radiographs"
# SEGMENTATION_DIR = DATA_ROOT / "Segmentation"/"Segmentation"
# DESCRIPTIONS_FILE = DATA_ROOT / "Expert" / "Expert" /"expert.json"   # or .json / .xlsx
# -------------------------------------------------------------------------------
DATA_ROOT = Path(r"./data/the-tufts-dental-database-2022")
RADIOGRAPHS_DIR = DATA_ROOT / "Radiographs" / "Radiographs"
SEGMENTATION_DIR = DATA_ROOT / "Segmentation" / "Segmentation"
DESCRIPTIONS_FILE = DATA_ROOT / "Expert" / "Expert" / "expert.json"

# Where all intermediate + final artifacts get written.
# Every stage reads/writes ONLY through these paths, which is what lets the
# image stage and the text stage stay fully decoupled.
OUTPUT_DIR = Path("./outputs")
IMAGE_FEATURES_CSV = OUTPUT_DIR / "image_features.csv"          # stage 1 output
TEXT_LOOKUP_CSV = OUTPUT_DIR / "text_lookup_table.csv"           # stage 2 output
FUSED_DATASET_CSV = OUTPUT_DIR / "fused_dataset.csv"             # stage 3 input built from the two above
MODEL_DIR = OUTPUT_DIR / "models"

IMG_SIZE = 224                 # ResNet input size
BATCH_SIZE = 16
NUM_EPOCHS_FINETUNE = 10
RANDOM_STATE = 42

for p in [OUTPUT_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
