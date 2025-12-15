%%writefile energy_snn_vs_llm.py
# -*- coding: utf-8 -*-
"""
energy_snn_vs_llm.py
1-file runnable in Colab/Jupyter/CLI.

Features:
- SNN (LIF) text classifier with cheap hash vectorizer
- DistilBERT baseline (optional), with optional quick train
- NVML energy measurement (GPU) via pynvml (fallback if unavailable)
- Wake-on-SNN Hybrid: call BERT only when SNN confidence < wake_thr
- Threshold sweep + plot: Accuracy vs Energy(J/sample) with wake-rate color
- Ignores Jupyter injected args like: -f /.../kernel.json
- Fixes HF datasets indexing (numpy.int64 -> int)

Notes:
- NVML energy is an approximation (power * elapsed), but works well for relative comparisons.
- If you see: "pynvml deprecated" -> optional; install nvidia-ml-py if you want.
"""

import os, time, random, argparse, math, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional NVML
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# ----------------------------
# Utils
# ----------------------------
def now_ms() -> float:
    return time.time() * 1000.0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(requested: str) -> torch.device:
    requested = (requested or "").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ----------------------------
# GPU energy meter (NVML)
# ----------------------------
class GpuEnergyMeter:
    """
    Approximate energy by reading GPU power at stop and multiplying by elapsed.
    Returns Joules, or None if not available.
    """
    def __init__(self, device: torch.device, gpu_index: int = 0):
        self.device = device
        self.gpu_index = gpu_index
        self._ok = False
        self._t0 = None
        self._handle = None

        if self.device.type == "cuda" and _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._ok = True
            except Exception:
                self._ok = False

    def start(self):
        if not self._ok:
            self._t0 = None
            return
        self._t0 = time.time()
        _ = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # warm

    def stop_joules(self) -> Optional[float]:
        if not self._ok or self._t0 is None:
            return None
        t1 = time.time()
        dt = max(0.0, t1 - self._t0)
        p_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
        p_w = p_mw / 1000.0
        return float(p_w * dt)

# ----------------------------
# Dataset
# ----------------------------
def load_text_dataset(name: str):
    """
    Supported:
      - ag_news (4 classes)
    Returns lists: train_texts, train_labels, test_texts, test_labels, n_classes
    """
    from datasets import load_dataset

    name = (name or "ag_news").lower()
    if name == "ag_news":
        ds = load_dataset("ag_news")
        train_texts = list(ds["train"]["text"])
        train_labels = list(ds["train"]["label"])
        test_texts  = list(ds["test"]["text"])
        test_labels = list(ds["test"]["label"])
        return train_texts, train_labels, test_texts, test_labels, 4

    raise ValueError(f"Unsupported dataset: {name}")

# ----------------------------
# Hash vectorizer
# ----------------------------
@dataclass
class HashVectorizer:
    dim: int = 8192
    seed: int = 123

    def featurize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        # Bag-of-words with hashing trick
        x = torch.zeros((len(texts), self.dim), device=device, dtype=torch.float32)
        for i, t in enumerate(texts):
            toks = t.lower().split()
            for w in toks[:256]:
                h = (hash((w, self.seed)) % self.dim)
                x[i, h] += 1.0
        x = x / (x.sum(dim=1, keepdim=True) + 1e-6)
        return x

# ----------------------------
# SNN (LIF) classifier
# ----------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, thr):
        out = (u >= thr).to(u.dtype)
        ctx.save_for_backward(u)
        ctx.thr = thr
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        thr = ctx.thr
        x = (u - thr).abs()
        # triangular surrogate
        grad = (x < 1.0).to(u.dtype) * (1.0 - x)
        return grad_out * grad, None

def spike(u, thr: float):
    return SpikeFn.apply(u, thr)

class SNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, steps: int = 20, beta: float = 0.9, thr: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True)
        self.steps = steps
        self.beta = beta
        self.thr = thr

    def forward(self, x):
        B = x.size(0)
        device = x.device
        u1 = torch.zeros((B, self.fc1.out_features), device=device)
        u2 = torch.zeros((B, self.fc2.out_features), device=device)

        out_sum = torch.zeros((B, self.fc2.out_features), device=device)
        spk_sum = 0.0

        # precompute current to speed a bit
        i1 = self.fc1(x)

        for _ in range(self.steps):
            u1 = self.beta * u1 + i1
            s1 = spike(u1, self.thr)
            u1 = u1 * (1.0 - s1)

            u2 = self.beta * u2 + self.fc2(s1)
            s2 = spike(u2, self.thr)
            u2 = u2 * (1.0 - s2)

            out_sum += s2
            spk_sum += s1.sum(dim=1).float().mean().item()

        logits = out_sum / float(self.steps)
        mean_spikes = spk_sum / float(self.steps)
        return logits, mean_spikes

# ----------------------------
# SNN training / eval / inference utils
# ----------------------------
def train_snn_one_epoch(model: nn.Module, vec: HashVectorizer, texts, labels, device, batch_size=128, lr=5e-4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idx = np.random.permutation(len(texts))

    total_loss = 0.0
    steps = 0
    for i in range(0, len(idx), batch_size):
        j = idx[i:i+batch_size].tolist()  # numpy.int64 -> python list[int]
        bt = [texts[int(k)] for k in j]
        by = torch.tensor([labels[int(k)] for k in j], device=device, dtype=torch.long)

        x = vec.featurize(bt, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, by)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)

@torch.no_grad()
def eval_snn(model: nn.Module, vec: HashVectorizer, texts, labels, device, batch_size=128):
    model.eval()
    correct = 0
    total = 0
    spikes = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        by = torch.tensor(labels[i:i+batch_size], device=device, dtype=torch.long)
        x = vec.featurize(bt, device)
        logits, mspk = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == by).sum().item())
        total += int(by.numel())
        spikes.append(float(mspk))
    return 100.0 * correct / max(1, total), float(np.mean(spikes))

@torch.no_grad()
def infer_snn_forward_only(model: nn.Module, vec: HashVectorizer, texts, device, batch_size=128):
    """
    Returns:
      logits_cpu: torch.FloatTensor [N, C] on CPU
      mean_spikes: float
    """
    model.eval()
    all_logits = []
    spikes = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        x = vec.featurize(bt, device)
        logits, mspk = model(x)
        all_logits.append(logits.detach().float().cpu())
        spikes.append(float(mspk))
    logits_cpu = torch.cat(all_logits, dim=0)
    return logits_cpu, float(np.mean(spikes))

def logits_to_conf_pred(logits_cpu: torch.Tensor):
    """
    logits_cpu: [N,C] on CPU
    Returns:
      conf: [N] float (max softmax prob)
      pred: [N] long
    """
    probs = torch.softmax(logits_cpu, dim=1)
    conf, pred = probs.max(dim=1)
    return conf, pred

# ----------------------------
# DistilBERT baseline (optional)
# ----------------------------
@torch.no_grad()
def infer_distilbert_pred(bert, tok, texts, device, batch_size=16, max_len=128):
    bert.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        enc = tok(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = bert(**enc).logits
        preds.append(out.argmax(dim=1).detach().cpu())
    return torch.cat(preds, dim=0) if len(preds) else torch.empty((0,), dtype=torch.long)

@torch.no_grad()
def eval_distilbert_acc(bert, tok, texts, labels, device, batch_size=16, max_len=128):
    pred = infer_distilbert_pred(bert, tok, texts, device, batch_size=batch_size, max_len=max_len)
    y = torch.tensor(labels, dtype=torch.long)
    return 100.0 * float((pred == y).float().mean().item())

def quick_train_distilbert(bert, tok, texts, labels, device,
                           steps=200, batch_size=16, max_len=128, lr=2e-5):
    """
    Very small quick training to make BERT baseline meaningful.
    """
    bert.train()
    opt = torch.optim.AdamW(bert.parameters(), lr=lr)

    n = len(texts)
    # sample indices each step (fast)
    for step in range(steps):
        idx = np.random.randint(0, n, size=(batch_size,))
        bt = [texts[int(i)] for i in idx]
        by = torch.tensor([labels[int(i)] for i in idx], device=device, dtype=torch.long)

        enc = tok(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        out = bert(**enc).logits
        loss = F.cross_entropy(out, by)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        opt.step()

    return float(loss.item())

# ----------------------------
# Hybrid + Sweep + Plot
# ----------------------------
def run_hybrid_once(snn, vec, bert, tok, test_texts, test_labels, device,
                    snn_batch, bert_batch, bert_max_len, wake_thr: float):
    # System-1: SNN forward only (this is real always-on cost)
    meter_snn = GpuEnergyMeter(device=device, gpu_index=0)
    meter_snn.start()
    t0 = now_ms()

    logits_cpu, _mspk = infer_snn_forward_only(snn, vec, test_texts, device, batch_size=snn_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    time_snn = (now_ms() - t0) / 1000.0
    energy_snn = meter_snn.stop_joules()

    conf, pred = logits_to_conf_pred(logits_cpu)
    true = torch.tensor(test_labels, dtype=torch.long)
    wake_mask = (conf < float(wake_thr))
    wake_idx = wake_mask.nonzero(as_tuple=True)[0].tolist()
    wake_rate = float(wake_mask.float().mean().item())  # 0..1

    # System-2: BERT only for woke samples
    meter_bert = GpuEnergyMeter(device=device, gpu_index=0)
    meter_bert.start()
    t1 = now_ms()

    if len(wake_idx) > 0:
        woke_texts = [test_texts[int(i)] for i in wake_idx]
        bert_pred = infer_distilbert_pred(
            bert, tok, woke_texts, device,
            batch_size=bert_batch, max_len=bert_max_len
        )
    else:
        bert_pred = torch.empty((0,), dtype=torch.long)

    if device.type == "cuda":
        torch.cuda.synchronize()
    time_bert = (now_ms() - t1) / 1000.0
    energy_bert = meter_bert.stop_joules()

    final_pred = pred.clone()
    if len(wake_idx) > 0:
        final_pred[wake_idx] = bert_pred.cpu()

    acc = float((final_pred == true).float().mean().item())  # 0..1

    total_j = None
    j_per_sample = None
    if (energy_snn is not None) and (energy_bert is not None):
        total_j = float(energy_snn + energy_bert)
        j_per_sample = total_j / max(1, len(test_texts))

    return {
        "wake_thr": float(wake_thr),
        "wake_rate": wake_rate,
        "acc": acc,
        "time_snn": float(time_snn),
        "time_bert": float(time_bert),
        "energy_snn_j": None if energy_snn is None else float(energy_snn),
        "energy_bert_j": None if energy_bert is None else float(energy_bert),
        "total_j": total_j,
        "j_per_sample": j_per_sample,
    }

def sweep_wake_thresholds(snn, vec, bert, tok, test_texts, test_labels, device,
                          snn_batch=128, bert_batch=16, bert_max_len=128,
                          thrs=None, save_json_path="wake_sweep.json"):
    if thrs is None:
        thrs = [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]

    rows = []
    for thr in thrs:
        r = run_hybrid_once(
            snn, vec, bert, tok,
            test_texts, test_labels, device,
            snn_batch, bert_batch, bert_max_len, float(thr)
        )
        rows.append(r)
        print(f"[SWEEP] thr={thr:.3f} wake={r['wake_rate']*100:.2f}% "
              f"acc={r['acc']*100:.2f}% J/sample={r['j_per_sample']:.6f}")

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {save_json_path}")
    return rows

def plot_tradeoff(rows, title="Accuracy vs Energy (Wake-on-SNN)"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skip plot.")
        return

    xs = [r["j_per_sample"] for r in rows]
    ys = [r["acc"] * 100.0 for r in rows]
    cs = [r["wake_rate"] * 100.0 for r in rows]
    labels = [f"{r['wake_thr']:.3f}" for r in rows]

    plt.figure()
    sc = plt.scatter(xs, ys, c=cs)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Energy (J / sample)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label("Wake rate (%)")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Args / Main
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--limit_train", type=int, default=20000)
    p.add_argument("--limit_test", type=int, default=2000)

    p.add_argument("--vec_dim", type=int, default=8192)
    p.add_argument("--snn_hidden", type=int, default=512)
    p.add_argument("--snn_steps", type=int, default=20)
    p.add_argument("--snn_epochs", type=int, default=5)
    p.add_argument("--snn_lr", type=float, default=5e-4)
    p.add_argument("--snn_batch", type=int, default=128)

    p.add_argument("--run_bert", action="store_true")
    p.add_argument("--bert_train", action="store_true")
    p.add_argument("--bert_train_steps", type=int, default=200)
    p.add_argument("--bert_batch", type=int, default=16)
    p.add_argument("--bert_max_len", type=int, default=128)
    p.add_argument("--bert_lr", type=float, default=2e-5)

    p.add_argument("--run_hybrid", action="store_true")
    p.add_argument("--wake_thr", type=float, default=0.34)

    p.add_argument("--sweep", action="store_true",
                   help="sweep multiple wake thresholds and plot tradeoff")
    p.add_argument("--sweep_thrs", type=str, default="0.30,0.32,0.34,0.35,0.36,0.37,0.38,0.39")

    return p

def main():
    args, _unknown = build_argparser().parse_known_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"🚀 device={device}")

    # dataset
    train_texts, train_labels, test_texts, test_labels, n_classes = load_text_dataset(args.dataset)
    train_texts = train_texts[:args.limit_train]
    train_labels = train_labels[:args.limit_train]
    test_texts  = test_texts[:args.limit_test]
    test_labels = test_labels[:args.limit_test]

    # SNN
    vec = HashVectorizer(dim=args.vec_dim, seed=args.seed)
    snn = SNNClassifier(in_dim=args.vec_dim, hidden=args.snn_hidden, n_classes=n_classes,
                        steps=args.snn_steps, beta=0.9, thr=1.0).to(device)

    # train SNN
    for ep in range(args.snn_epochs):
        t0 = now_ms()
        loss = train_snn_one_epoch(
            snn, vec, train_texts, train_labels, device,
            batch_size=args.snn_batch, lr=args.snn_lr
        )
        # quick "train-subset acc" (cheap sanity check)
        subset_n = min(2000, len(train_texts))
        sub_acc, sub_spk = eval_snn(
            snn, vec, train_texts[:subset_n], train_labels[:subset_n], device,
            batch_size=args.snn_batch
        )
        print(f"[SNN] ep {ep+1}/{args.snn_epochs} loss={loss:.4f} "
              f"train_subset_acc={sub_acc:.2f}% mean_spikes={sub_spk:.2f} time={(now_ms()-t0)/1000:.1f}s")

    # evaluate SNN on test
    test_acc, test_spk = eval_snn(snn, vec, test_texts, test_labels, device, batch_size=args.snn_batch)
    print(f"[SNN-CONF] test_acc={test_acc:.2f}% mean_spikes={test_spk:.2f}")

    # confidence stats
    logits_cpu, _ = infer_snn_forward_only(snn, vec, test_texts, device, batch_size=args.snn_batch)
    conf, _pred = logits_to_conf_pred(logits_cpu)
    q = torch.quantile(conf, torch.tensor([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]))
    q_map = {float(k): float(v) for k, v in zip([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99], q)}
    print(f"[SNN-CONF] min={float(conf.min()):.4f} mean={float(conf.mean()):.4f} max={float(conf.max()):.4f}")
    print(f"[SNN-CONF-QUANTILE] {q_map}")

    # energy SNN inference
    meter = GpuEnergyMeter(device=device, gpu_index=0)
    meter.start()
    t0 = now_ms()
    logits_cpu2, _ = infer_snn_forward_only(snn, vec, test_texts, device, batch_size=args.snn_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_snn = (now_ms() - t0) / 1000.0
    j_snn = meter.stop_joules()
    conf2, pred2 = logits_to_conf_pred(logits_cpu2)
    true2 = torch.tensor(test_labels, dtype=torch.long)
    acc_snn = 100.0 * float((pred2 == true2).float().mean().item())
    print(f"[SNN] infer_acc={acc_snn:.2f}% time={dt_snn:.3f}s energy_j={j_snn}")
    if j_snn is not None:
        print(f"[SNN] J/sample={j_snn/len(test_texts):.6f}")

    # BERT optional
    bert = None
    tok = None
    j_bert = None
    if args.run_bert:
        from transformers import AutoTokenizer, DistilBertForSequenceClassification
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=n_classes
        ).to(device)

        if args.bert_train:
            tbt = now_ms()
            last_loss = quick_train_distilbert(
                bert, tok, train_texts, train_labels, device,
                steps=args.bert_train_steps, batch_size=args.bert_batch,
                max_len=args.bert_max_len, lr=args.bert_lr
            )
            print(f"[BERT-train] steps={args.bert_train_steps} last_loss={last_loss:.4f} time={(now_ms()-tbt)/1000:.1f}s")

        meter2 = GpuEnergyMeter(device=device, gpu_index=0)
        meter2.start()
        t1 = now_ms()
        acc_bert = eval_distilbert_acc(
            bert, tok, test_texts, test_labels, device,
            batch_size=args.bert_batch, max_len=args.bert_max_len
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_bert = (now_ms() - t1) / 1000.0
        j_bert = meter2.stop_joules()
        print(f"[BERT] infer_acc={acc_bert:.2f}% time={dt_bert:.3f}s energy_j={j_bert}")
        if j_bert is not None:
            print(f"[BERT] J/sample={j_bert/len(test_texts):.6f}")
        if (j_snn is not None) and (j_bert is not None) and j_snn > 0:
            print(f"[RATIO] BERT/SNN energy ≈ {j_bert/j_snn:.2f}x")

    # Hybrid single run
    if args.run_hybrid:
        if bert is None or tok is None:
            raise RuntimeError("--run_hybrid requires --run_bert (BERT model/tokenizer needed).")

        print("========== Wake-on-SNN Hybrid ==========")
        wake_thr = float(args.wake_thr)
        r = run_hybrid_once(
            snn, vec, bert, tok,
            test_texts, test_labels, device,
            snn_batch=args.snn_batch,
            bert_batch=args.bert_batch,
            bert_max_len=args.bert_max_len,
            wake_thr=wake_thr
        )
        print(f"[HYBRID] wake_thr={r['wake_thr']:.3f}  wake_rate={r['wake_rate']*100:.2f}%")
        print(f"[HYBRID] acc={r['acc']*100:.2f}%")
        print(f"[HYBRID] time_snn={r['time_snn']:.3f}s  time_bert={r['time_bert']:.3f}s  total={(r['time_snn']+r['time_bert']):.3f}s")
        print(f"[HYBRID] energy_snn_j={r['energy_snn_j']}  energy_bert_j={r['energy_bert_j']}")
        if r["total_j"] is not None:
            print(f"[HYBRID] total_j={r['total_j']:.6f}  J/sample={r['j_per_sample']:.6f}")
            if j_bert is not None and r["total_j"] > 0:
                print(f"[SAVING] BERT(always)/HYBRID energy ≈ {j_bert/r['total_j']:.2f}x")

    # Sweep + plot
    if args.sweep:
        if bert is None or tok is None:
            raise RuntimeError("--sweep requires --run_bert (BERT needed).")

        thrs = [float(x.strip()) for x in args.sweep_thrs.split(",") if x.strip()]
        rows = sweep_wake_thresholds(
            snn, vec, bert, tok,
            test_texts, test_labels, device,
            snn_batch=args.snn_batch,
            bert_batch=args.bert_batch,
            bert_max_len=args.bert_max_len,
            thrs=thrs,
            save_json_path="wake_sweep.json"
        )
        plot_tradeoff(rows)

if __name__ == "__main__":
    main())