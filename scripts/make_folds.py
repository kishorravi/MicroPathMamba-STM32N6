"""Create the stratified 3-fold split once and reuse it for every model (including ATLAS-Net).

python scripts/make_folds.py --data NCT-CRC-HE-100K --out folds.csv --seed 123

Note for the paper: NCT-CRC-HE-100K file names carry no patient IDs, so this split is
patch-level. Patches from one patient can fall in both train and validation folds, which
inflates internal CV. Treat CRC-VAL-HE-7K (different patients) as the primary result.
"""
import argparse
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import list_folder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="folds.csv")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--seed", type=int, default=123)
a = ap.parse_args()

df = list_folder(a.data)
df["fold"] = -1
for f, (_, idx) in enumerate(StratifiedKFold(a.k, shuffle=True, random_state=a.seed).split(df, df.label)):
    df.loc[df.index[idx], "fold"] = f
df.to_csv(a.out, index=False)
print(df.groupby(["fold", "label"]).size().unstack())
