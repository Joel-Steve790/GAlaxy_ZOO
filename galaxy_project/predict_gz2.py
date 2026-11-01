#!/usr/bin/env python3
"""Classify one or more galaxy images with a model trained by train_gz2.py.

  python predict_gz2.py --ckpt gz2_resnet50.pt path/to/galaxy.jpg [more.jpg ...]
"""
import argparse, torch
from PIL import Image
from torchvision import models, transforms as T
import torch.nn as nn

TFM = T.Compose([T.CenterCrop(256), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load(ckpt, dev):
    c = torch.load(ckpt, map_location=dev)
    n = len(c["classes"])
    if c["arch"] == "resnet50":
        m = models.resnet50(); m.fc = nn.Linear(m.fc.in_features, n)
    else:
        m = models.efficientnet_b0(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    m.load_state_dict(c["state_dict"]); return m.to(dev).eval(), c["classes"]


@torch.no_grad()
def predict(model, classes, path, dev):
    img = Image.open(path).convert("RGB")
    # light test-time augmentation: average over 4 flips/rotations (galaxies have no 'up')
    views = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.rotate(90), img.rotate(180)]
    x = torch.stack([TFM(v) for v in views]).to(dev)
    return dict(zip(classes, model(x).softmax(1).mean(0).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default="gz2_resnet50.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load(a.ckpt, dev)
    for p in a.images:
        probs = predict(model, classes, p, dev)
        top = max(probs, key=probs.get)
        print(f"{p}: {top} ({probs[top]*100:.1f}%)  |  " +
              "  ".join(f"{k}={v*100:.1f}%" for k, v in probs.items()))
