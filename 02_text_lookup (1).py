# %% [markdown]
# # Stage 2 — Text lookup table
#
# Builds the keyword lookup table from the expert free-text descriptions
# (the "S01_a_Impacted 38: Impaction with 38" style lines).
#
# This file never imports anything from Stage 1 (tooth_detector.py,
# resnet_features.py) and Stage 1 never imports anything from here — that's
# what lets you add this stage at any point without editing 01_image_pipeline.py.
#
# Output: `outputs/text_lookup_table.csv`

# %%
from config import DESCRIPTIONS_FILE, TEXT_LOOKUP_CSV
from src.text_lookup import build_text_lookup_table

# %% [markdown]
# ## 1. Parse the raw descriptions file into a tidy table

# %%
lookup_table = build_text_lookup_table(DESCRIPTIONS_FILE)
print(f"Parsed {len(lookup_table)} findings across {lookup_table['image_id'].nunique()} images")
lookup_table.head(15)

# %% [markdown]
# ## 2. Sanity-check keyword coverage

# %%
all_keywords = lookup_table["keywords"].dropna().str.split("|").explode()
print(all_keywords.value_counts())

# %% [markdown]
# ## 3. Save

# %%
lookup_table.to_csv(TEXT_LOOKUP_CSV, index=False)
print(f"Wrote TDD training lookup table -> {TEXT_LOOKUP_CSV}")

# %% [markdown]
# ## 4. Build Lookup Table for Test Descriptions (S01 - S06)
#
# Here we demonstrate generating the keyword lookup table for the separate text
# descriptions uploaded for the test images (S01_a, S01_b, etc.).

# %%
from pathlib import Path
test_descriptions_path = Path("data/test_descriptions.txt")

if test_descriptions_path.exists():
    test_lookup_table = build_text_lookup_table(test_descriptions_path)
    print(f"\nParsed {len(test_lookup_table)} test anomaly descriptions across {test_lookup_table['image_id'].nunique()} images:")
    print(test_lookup_table[["image_id", "finding_id", "tooth_number", "finding_phrase", "keywords"]])
    
    test_csv_path = Path("outputs/test_descriptions_lookup.csv")
    test_lookup_table.to_csv(test_csv_path, index=False)
    print(f"\nWrote test descriptions lookup table -> {test_csv_path}")
else:
    print(f"Test descriptions file not found at {test_descriptions_path}")
# %%

