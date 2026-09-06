"""Evaluate timing probabilities on the checkpoint's held-out split; no training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from policy_v1.data import Windows, collate
from policy_v1.train import load_checkpoint, move
from .model import Config, Policy


THRESHOLDS = (0.0001, 0.0003, 0.001, 0.003, 0.005, 0.006,
              0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5)


def probability_report(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("nonempty matching 1D probabilities/labels required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probabilities")
    positives = int(y.sum())
    negatives = len(y) - positives

    def distribution(values):
        if not len(values):
            return None
        return {
            "count": len(values), "mean": float(values.mean()),
            "quantiles": dict(zip(
                ("min", "p25", "p50", "p75", "p90", "p99", "max"),
                np.quantile(values, [0, .25, .5, .75, .9, .99, 1]).tolist())),
        }

    # Group tied scores before computing AP/AUC; a constant predictor must not
    # receive artificial ranking credit from the order of the input records.
    order = np.argsort(-p, kind="stable")
    scores, ranked = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(p)-1]
    tp = np.cumsum(ranked)[ends].astype(np.float64)
    fp = ends + 1 - tp
    ap = auc = None
    if positives:
        recall = tp / positives
        precision = tp / (ends+1)
        ap = float(np.sum(np.diff(np.r_[0., recall])*precision))
        if negatives:
            x = np.r_[0., fp/negatives]
            z = np.r_[0., recall]
            auc = float(np.sum(np.diff(x)*(z[1:]+z[:-1])/2))
    thresholds = []
    for threshold in THRESHOLDS:
        predicted = p > threshold  # matches the trainer's strict p > 0.5
        true_positive = int((predicted & y).sum())
        false_positive = int((predicted & ~y).sum())
        n_predicted = true_positive + false_positive
        thresholds.append({
            "threshold": threshold, "predicted_actions": n_predicted,
            "tp": true_positive, "fp": false_positive,
            "fn": positives-true_positive,
            "precision": true_positive/n_predicted if n_predicted else None,
            "recall": true_positive/positives if positives else None,
            "false_positive_rate": false_positive/negatives if negatives else None,
            "predicted_action_rate": n_predicted/len(y),
        })
    return {
        "valid_frames": len(y), "actual_actions": positives,
        "actual_action_rate": positives/len(y),
        "always_wait_accuracy": negatives/len(y),
        "mean_predicted_probability": float(p.mean()),
        "average_precision": ap,
        "constant_score_ap_baseline": positives/len(y) if positives else None,
        "ap_lift_over_prevalence": ap/(positives/len(y)) if positives else None,
        "roc_auc": auc,
        "action_probabilities": distribution(p[y]),
        "wait_probabilities": distribution(p[~y]),
        "thresholds": thresholds,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--runs", type=Path, default=Path("/root/autodl-tmp/runs"))
    p.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/expert-dataset/native-bc-v1"))
    p.add_argument("--cache", type=Path, default=Path("/root/autodl-tmp/policy-v1-cache"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-smoke", action="store_true")
    return p


def run(args):
    if min(args.batch_size, args.batches, args.cpu_threads) < 1 or args.workers < 0:
        raise ValueError("positive batch counts/threads and nonnegative workers required")
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list(args.runs.glob("hokoff-lstm-check*/last.pt"))
        if not candidates:
            raise FileNotFoundError("No HoKoff checkpoint found; pass --checkpoint PATH")
        checkpoint = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    saved = load_checkpoint(checkpoint)
    if saved["config"].get("architecture") != "hokoff_cr_lstm_v1":
        raise ValueError("requires a hokoff_cr_lstm_v1 checkpoint")
    contract = saved["contract"]
    split = contract["val_split"]
    if split == contract["train_split"] or split == "test":
        raise ValueError("timing diagnostics require a non-test held-out split")
    output_path = args.output or checkpoint.parent / (
        "timing-eval-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    if output_path.exists():
        raise FileExistsError(output_path)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU diagnostics require --device cpu")
    config = Config(**saved["config"])
    dataset = Windows(args.data, args.cache, split,
                      targets=contract["targets"], frame_window=config.frame_window,
                      event_window=1)
    if dataset.index["manifest_sha256"] != contract["manifest_sha256"]:
        raise ValueError("checkpoint and dataset manifest differ")
    if dataset.index["smoke_only"] and not args.allow_smoke:
        raise ValueError("synthetic data requires --allow-smoke")
    if not len(dataset):
        raise ValueError("empty held-out split")
    # Deterministic windows spread over the held-out split, rather than just
    # its first shards. No frame balancing: retain the natural WAIT exposure.
    n_windows = min(len(dataset), args.batch_size*args.batches)
    indices = random.Random(args.seed).sample(range(len(dataset)), n_windows)
    loader = DataLoader(dataset, sampler=indices, batch_size=args.batch_size,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(args.seed))
    model = Policy(config).to(device).eval()
    model.load_state_dict(saved["model"])
    # This evaluator deliberately uses FP32 to avoid FP16 rounding in low
    # probabilities and tied-score ranking. No optimizer is constructed.
    print(json.dumps({"phase": "timing_eval_start", "checkpoint": str(checkpoint),
                      "checkpoint_step": saved["step"], "split": split,
                      "windows": n_windows, "precision": "fp32"}), flush=True)
    probabilities, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            b = move(batch, device)
            valid = b["frame_mask"] & b["loss_mask"] & b["timing_label_mask"]
            logits = model(b)["timing"][valid].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite timing logits")
            probabilities.append(logits.sigmoid().cpu().numpy())
            labels.append(b["play_now"][valid].cpu().numpy())
            if i % 50 == 0:
                print(json.dumps({"phase": "timing_eval_progress", "batches": i}), flush=True)
    report = probability_report(np.concatenate(probabilities), np.concatenate(labels))
    report.update(checkpoint=str(checkpoint), checkpoint_step=saved["step"],
                  split=split, windows=n_windows, sampling="uniform_windows_without_replacement",
                  seed=args.seed, precision="fp32", metrics_weighting="unweighted_valid_frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"phase": "timing_eval", "report_path": str(output_path), **report}), flush=True)
    return report


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
