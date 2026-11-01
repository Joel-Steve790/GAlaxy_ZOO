#!/usr/bin/env python3
"""Galaxy Zoo 2 morphology classifier (PyTorch, transfer learning).

Expected layout:
    archive/gz2_filename_mapping.csv      (objid, sample, asset_id)
    archive/gz2_hart16.csv                (labels: dr7objid, gz2_class, ...)  <-- you must add this
    archive/images_gz2/images/{asset_id}.jpg

Usage:
    python train_gz2.py --root archive --labels archive/gz2_hart16.csv
    python train_gz2.py --root archive --labels archive/gz2_hart16.csv --arch efficientnet_b0 --epochs 15
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

CLASSES = ["smooth_round", "smooth_inbetween", "smooth_cigar",
           "edge_on", "spiral", "barred_spiral"]


def gz2_class_to_label(c: str):
    """Collapse the detailed Hart+16 gz2_class string into 6 coarse classes."""
    if not isinstance(c, str) or c.startswith("A"):    # artifacts / unknown -> drop
        return None
    if c.startswith("Er"): return 0
    if c.startswith("Ei"): return 1
    if c.startswith("Ec"): return 2
    if c.startswith("Se"): return 3                     # edge-on disk
    if c.startswith("SB"): return 5                     # barred spiral
    if c.startswith("S"):  return 4                     # spiral
    return None


def build_table(root, labels_csv):
    m = pd.read_csv(os.path.join(root, "gz2_filename_mapping.csv"))
    lab = pd.read_csv(labels_csv)
    lab = lab[["dr7objid", "gz2_class"]].rename(columns={"dr7objid": "objid"})
    df = m.merge(lab, on="objid", how="inner")
    df["label"] = df["gz2_class"].map(gz2_class_to_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    img_dir = os.path.join(root, "images_gz2", "images")
    df["path"] = df["asset_id"].astype(str).map(lambda a: os.path.join(img_dir, a + ".jpg"))
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
    return df


class GZ2(Dataset):
    def __init__(self, df, tfm):
        self.paths, self.y, self.tfm = df["path"].values, df["label"].values, tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), int(self.y[i])


def make_model(arch, n_classes):
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev, non_blocking=True)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="archive")
    ap.add_argument("--labels", required=True, help="path to gz2_hart16.csv")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="subsample N rows for a quick test")
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = build_table(a.root, a.labels)
    if a.limit: df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    print(f"{len(df)} labelled images\n", df["label"].map(dict(enumerate(CLASSES))).value_counts())

    tr, tmp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=0)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=0)

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Galaxies are centred and rotation-invariant: crop the middle, flip/rotate freely.
    train_tfm = T.Compose([T.CenterCrop(256), T.RandomRotation(180), T.RandomHorizontalFlip(),
                           T.CenterCrop(224), T.ToTensor(), norm])
    eval_tfm = T.Compose([T.CenterCrop(256), T.CenterCrop(224), T.ToTensor(), norm])
    mk = lambda d, t, sh: DataLoader(GZ2(d, t), batch_size=a.bs, shuffle=sh,
                                     num_workers=a.workers, pin_memory=True)
    dl_tr, dl_va, dl_te = mk(tr, train_tfm, True), mk(va, eval_tfm, False), mk(te, eval_tfm, False)

    model = make_model(a.arch, len(CLASSES)).to(dev)
    counts = np.bincount(tr["label"], minlength=len(CLASSES))
    w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)                 # handles class imbalance
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
        yv, pv = evaluate(model, dl_va, dev)
        f1 = f1_score(yv, pv, average="macro")
        print(f"ep {ep+1}/{a.epochs} loss {tot/len(dl_tr):.3f} val acc {accuracy_score(yv, pv):.4f} "
              f"macroF1 {f1:.4f} ({time.time()-t0:.0f}s)")
        if f1 > best:
            best = f1; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    yt, pt = evaluate(model, dl_te, dev)
    print(f"\nTEST  acc {accuracy_score(yt, pt):.4f}  macro-F1 {f1_score(yt, pt, average='macro'):.4f}")
    print(classification_report(yt, pt, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true):\n", confusion_matrix(yt, pt))


if __name__ == "__main__":
    main()
