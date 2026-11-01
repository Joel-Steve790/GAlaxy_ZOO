#!/usr/bin/env python3
"""Galaxy Zoo 2 -> 3-class galaxy classifier (PyTorch transfer learning).

Classes
  0 smooth_elliptical        smooth, featureless (round/spherical, in-between, cigar ellipticals)
  1 disk_with_spiral_arms    disk galaxy with visible spiral arms
  2 disk_without_spiral_arms disk galaxy, no visible arms (includes edge-on disks)

Layout (run from GAlaxy_ZOO/galaxy_project):
  archive/gz2_filename_mapping.csv
  archive/gz2_hart16.csv
  archive/images_gz2/images/{asset_id}.jpg

  python train_gz2.py --root archive --limit 5000 --epochs 2     # quick sanity check
  python train_gz2.py --root archive --epochs 10                 # full run
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

CLASSES = ["smooth_elliptical", "disk_with_spiral_arms", "disk_without_spiral_arms"]
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TFM = T.Compose([T.CenterCrop(256), T.CenterCrop(224), T.ToTensor(), NORM])


def assign_labels(lab, t):
    """Confident labels from Hart+16 debiased/weighted vote fractions (threshold t)."""
    p = "_weighted_fraction"
    g = lambda c: lab[c + p]
    smooth = g("t01_smooth_or_features_a01_smooth")
    disk = g("t01_smooth_or_features_a02_features_or_disk")
    edge_yes, edge_no = g("t02_edgeon_a04_yes"), g("t02_edgeon_a05_no")
    spiral, no_spiral = g("t04_spiral_a08_spiral"), g("t04_spiral_a09_no_spiral")
    y = pd.Series(-1, index=lab.index)
    y[smooth >= t] = 0
    y[(disk >= t) & (edge_no >= t) & (spiral >= t)] = 1
    # edge-on disks: arms can't be seen, so GZ2 never asks the spiral question for them
    y[(disk >= t) & ((edge_yes >= t) | ((edge_no >= t) & (no_spiral >= t)))] = 2
    return y


def build_table(root, labels_csv, t):
    m = pd.read_csv(os.path.join(root, "gz2_filename_mapping.csv"))
    lab = pd.read_csv(labels_csv)
    lab["label"] = assign_labels(lab, t)
    lab = lab[lab["label"] >= 0][["dr7objid", "label"]].rename(columns={"dr7objid": "objid"})
    df = m.merge(lab, on="objid", how="inner").drop_duplicates("objid")  # 1 image per galaxy
    img_dir = os.path.join(root, "images_gz2", "images")
    df["path"] = df["asset_id"].astype(str).map(lambda a: os.path.join(img_dir, a + ".jpg"))
    return df[df["path"].map(os.path.exists)].reset_index(drop=True)


class GZ2(Dataset):
    def __init__(self, df, tfm):
        self.p, self.y, self.t = df["path"].values, df["label"].values, tfm
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return self.t(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


def make_model(arch, n):
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT); m.fc = nn.Linear(m.fc.in_features, n)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev, non_blocking=True)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="archive")
    ap.add_argument("--labels", default=None, help="default: <root>/gz2_hart16.csv")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--thresh", type=float, default=0.7, help="min vote fraction for a confident label")
    ap.add_argument("--max-per-class", type=int, default=40000, help="cap per class in TRAIN set (0=off)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = build_table(a.root, a.labels or os.path.join(a.root, "gz2_hart16.csv"), a.thresh)
    if a.limit: df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    print(f"{len(df)} labelled images found on disk")
    print(df["label"].map(dict(enumerate(CLASSES))).value_counts(), "\n")

    tr, tmp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=0)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=0)
    if a.max_per_class:  # tame the huge 'smooth' class, in train only (val/test stay realistic)
        tr = pd.concat([tr[tr["label"] == c].sample(min((tr["label"] == c).sum(), a.max_per_class),
                                                    random_state=0)
                        for c in sorted(tr["label"].unique())]).reset_index(drop=True)

    train_tfm = T.Compose([T.CenterCrop(256), T.RandomRotation(180), T.RandomHorizontalFlip(),
                           T.CenterCrop(224), T.ToTensor(), NORM])
    mk = lambda d, t, sh: DataLoader(GZ2(d, t), batch_size=a.bs, shuffle=sh,
                                     num_workers=a.workers, pin_memory=True)
    dl_tr, dl_va, dl_te = mk(tr, train_tfm, True), mk(va, EVAL_TFM, False), mk(te, EVAL_TFM, False)

    model = make_model(a.arch, len(CLASSES)).to(dev)
    cnt = np.bincount(tr["label"], minlength=len(CLASSES))
    w = torch.tensor(cnt.sum() / (len(CLASSES) * cnt), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best, ckpt = 0.0, f"gz2_{a.arch}.pt"
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        yv, pv = evaluate(model, dl_va, dev); f1 = f1_score(yv, pv, average="macro")
        print(f"ep {ep+1}/{a.epochs} loss {tot/len(dl_tr):.3f} val acc {accuracy_score(yv, pv):.4f} "
              f"macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best:
            best = f1
            torch.save({"arch": a.arch, "classes": CLASSES, "state_dict": model.state_dict()}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    yt, pt = evaluate(model, dl_te, dev)
    print(f"\nTEST acc {accuracy_score(yt, pt):.4f}  macro-F1 {f1_score(yt, pt, average='macro'):.4f}")
    print(classification_report(yt, pt, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true):\n", confusion_matrix(yt, pt))
    print(f"\nSaved best model to {ckpt}")


if __name__ == "__main__":
    main()
