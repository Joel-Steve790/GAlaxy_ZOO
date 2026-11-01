import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset paths
MAPPING = "archive/gz2_filename_mapping.csv"
CATALOGUE = "archive/gz2_hart16.csv"
IMAGE_DIR = Path("archive/images_gz2/images")

OUTPUT_DIR = Path("prepared")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.80
MIN_VOTES = 20

# Read galaxy IDs as strings to avoid losing integer precision.
mapping = pd.read_csv(MAPPING, dtype={"objid": str, "sample": str})
catalogue = pd.read_csv(
    CATALOGUE,
    dtype={"dr7objid": str, "sample": str}
)

print("Mapping rows:", len(mapping))
print("Catalogue rows:", len(catalogue))

# Remove duplicate mapping/catalogue records.
mapping = mapping.drop_duplicates(["objid", "sample"])
catalogue = catalogue.drop_duplicates(["dr7objid", "sample"])

# Connect galaxy IDs and sample categories.
df = mapping.merge(
    catalogue,
    left_on=["objid", "sample"],
    right_on=["dr7objid", "sample"],
    how="inner"
)

print("Matched catalogue records:", len(df))

# Convert image asset IDs into filenames.
df["asset_id"] = pd.to_numeric(df["asset_id"], errors="coerce")
df = df.dropna(subset=["asset_id"]).copy()
df["asset_id"] = df["asset_id"].astype(int)
df["image_filename"] = df["asset_id"].astype(str) + ".jpg"

# Keep only images that actually exist.
df = df[
    df["image_filename"].map(
        lambda filename: (IMAGE_DIR / filename).is_file()
    )
].copy()

print("Records with existing images:", len(df))

# Galaxy Zoo 2 debiased morphology columns.
smooth = "t01_smooth_or_features_a01_smooth_debiased"
features = "t01_smooth_or_features_a02_features_or_disk_debiased"
spiral = "t04_spiral_a08_spiral_debiased"
no_spiral = "t04_spiral_a09_no_spiral_debiased"

numeric_columns = [
    smooth, features, spiral, no_spiral, "total_votes"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)

df = df.dropna(subset=numeric_columns)
df = df[df["total_votes"] >= MIN_VOTES].copy()

# Conservative classification rules.
# 0 = Smooth
# 1 = Spiral
# 2 = Disk without spiral structure

smooth_mask = (
    (df[smooth] >= THRESHOLD) &
    (df[features] <= 1 - THRESHOLD)
)

spiral_mask = (
    (df[features] >= THRESHOLD) &
    (df[spiral] >= THRESHOLD)
)

disk_no_spiral_mask = (
    (df[features] >= THRESHOLD) &
    (df[no_spiral] >= THRESHOLD)
)

df["label"] = -1

df.loc[smooth_mask, "label"] = 0
df.loc[spiral_mask, "label"] = 1
df.loc[disk_no_spiral_mask, "label"] = 2

# Remove ambiguous examples.
df = df[df["label"] >= 0].copy()

# Avoid having the same galaxy in multiple splits.
df = df.drop_duplicates(subset=["objid"])

df["image_path"] = df["image_filename"].map(
    lambda name: str(IMAGE_DIR / name)
)

class_names = {
    0: "Smooth",
    1: "Spiral",
    2: "Disk_without_spiral"
}

df["class_name"] = df["label"].map(class_names)

print("\nFinal dataset:")
print(df["class_name"].value_counts())

if len(df) == 0:
    raise RuntimeError("No labelled images found. Check the data and paths.")

# 70% training, 15% validation, 15% testing.
train_val, test = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=train_val["label"]
)

# Save datasets.
train.to_csv(OUTPUT_DIR / "train.csv", index=False)
val.to_csv(OUTPUT_DIR / "val.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nDataset preparation completed!")
print("Training images:", len(train))
print("Validation images:", len(val))
print("Testing images:", len(test))

print("\nCSV files saved in:", OUTPUT_DIR)
