"""Make the synthetic doodles, train the reader on the CPU, and measure it on doodles of held-out cats.

Training doodles start from AFHQ's train cats, validation doodles from its test cats, so no validation picture
shares a cat's description or face layout with training. Half of each batch is mirrored, with the answer mirrored
the same way (CatAttributes.mirrored). Every head is scored against the trivial answer — always its most common
class — and the distractor check draws the same validation doodles again without the things beside the cat:
a reader that ignores them gives the same answer both times.

    python -m catdoodle.train --out runs/v0                 # data, training, report
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import HEADS, Reader, encode
from .sources import load_cats
from .synth import draw_doodle

_CATS = None


def _init():
    global _CATS
    _CATS = load_cats()


def _make(args):
    i, seed, distractors = args
    records, points, _ = _CATS
    d = draw_doodle(np.random.default_rng(seed), records[i], points[i], distractors=distractors)
    return d.image, encode(d.answer), encode(d.answer.mirrored())


def make_set(indices: np.ndarray, per_cat: int, base_seed: int, distractors: bool = True, workers: int = 6):
    jobs = [(int(i), base_seed + 1000 * int(i) + k, distractors) for i in indices for k in range(per_cat)]
    with Pool(workers, initializer=_init) as pool:
        out = pool.map(_make, jobs, chunksize=64)
    images = np.stack([o[0] for o in out])
    labels = {h: np.array([o[1][h] for o in out]) for h in HEADS}
    mirrored = {h: np.array([o[2][h] for o in out]) for h in HEADS}
    return images, labels, mirrored


def batches(images, labels, mirrored, size, rng, train):
    order = rng.permutation(len(images)) if train else np.arange(len(images))
    for s in range(0, len(order), size):
        idx = order[s:s + size]
        x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float() / 255
        y = {h: torch.from_numpy(labels[h][idx]) for h in HEADS}
        if train:                                   # mirror half the batch, answers included
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            x[flip] = x[flip].flip(-1)
            for h in HEADS:
                y[h] = torch.where(flip, torch.from_numpy(mirrored[h][idx]), y[h])
        yield x, y


def predict(model, images, size=256):
    model.eval()
    out = {h: [] for h in HEADS}
    with torch.no_grad():
        for s in range(0, len(images), size):
            x = torch.from_numpy(images[s:s + size]).permute(0, 3, 1, 2).float() / 255
            for h, logits in model(x).items():
                out[h].append(logits.argmax(1).numpy())
    return {h: np.concatenate(v) for h, v in out.items()}


def report(pred, labels, train_labels):
    rows = {}
    for h in HEADS:
        majority = np.bincount(train_labels[h], minlength=len(HEADS[h])).argmax()
        rows[h] = {"accuracy": float((pred[h] == labels[h]).mean()),
                   "majority_baseline": float((labels[h] == majority).mean()),
                   "majority_class": HEADS[h][majority],
                   "per_class_recall": {c: (float((pred[h][labels[h] == k] == k).mean())
                                            if (labels[h] == k).any() else None) for k, c in enumerate(HEADS[h])},
                   "count": {c: int((labels[h] == k).sum()) for k, c in enumerate(HEADS[h])}}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/v0"))
    ap.add_argument("--per-cat", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    _, _, split = load_cats()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    t0 = time.time()
    xtr, ytr, mtr = make_set(tr, a.per_cat, 1)
    xva, yva, _ = make_set(va, 4, 7_000_000)
    xva_clean, _, _ = make_set(va, 4, 7_000_000, distractors=False)
    print(f"data: {len(xtr)} train, {len(xva)} validation doodles in {time.time() - t0:.0f} s", flush=True)
    model = Reader()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(xtr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    rng = np.random.default_rng(a.seed)
    log, best = [], -1.0
    for epoch in range(a.epochs):
        model.train()
        t1, total, n = time.time(), 0.0, 0
        for x, y in batches(xtr, ytr, mtr, a.batch, rng, True):
            out = model(x)
            loss = sum(F.cross_entropy(out[h], y[h]) for h in HEADS) / len(HEADS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        pred = predict(model, xva)
        mean_acc = float(np.mean([(pred[h] == yva[h]).mean() for h in HEADS]))
        log.append({"epoch": epoch + 1, "train_loss": total / n, "val_mean_accuracy": mean_acc,
                    "seconds": round(time.time() - t1)})
        print(json.dumps(log[-1]), flush=True)
        if mean_acc > best:
            best = mean_acc
            torch.save(model.state_dict(), a.out / "reader.pt")
    model.load_state_dict(torch.load(a.out / "reader.pt"))
    pred, pred_clean = predict(model, xva), predict(model, xva_clean)
    same = np.mean([np.all([pred[h][i] == pred_clean[h][i] for h in HEADS]) for i in range(len(xva))])
    result = {"train_doodles": int(len(xtr)), "val_doodles": int(len(xva)), "epochs": log,
              "heads": report(pred, yva, ytr),
              "distractors": {"same_answer_with_and_without": float(same),
                              "per_head_same": {h: float((pred[h] == pred_clean[h]).mean()) for h in HEADS}}}
    (a.out / "report.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(a.out / "val_examples.npz", images=xva[:64], **{f"y_{h}": yva[h][:64] for h in HEADS},
                        **{f"p_{h}": pred[h][:64] for h in HEADS})
    for h, r in result["heads"].items():
        print(f"{h:18s} acc {r['accuracy']:.3f}  (always '{r['majority_class']}': {r['majority_baseline']:.3f})")
    print(f"distractors: same full answer with and without {same:.3f}")


if __name__ == "__main__":
    main()
