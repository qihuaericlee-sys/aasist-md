"""
Benchmark training and evaluation speed on Mandarin_flac + AASIST.
Measures real throughput on current hardware.
"""
import os, sys, time
sys.path.insert(0, '.')
from pathlib import Path
import json
import torch
import torch.nn as nn
from importlib import import_module
from torch.utils.data import DataLoader

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)

# ── Config ──────────────────────────────────────────────
config_path = "config/AASIST.conf"
with open(config_path, "r") as f:
    config = json.loads(f.read())

model_config = config["model_config"]
batch_size = config["batch_size"]

database_path = Path("../Mandarin_flac/")
protocol_dir = Path("/Users/qihualee/deepfake-audio/CFAD_cm_protocols_rock/CFAD_cm_protocols")

# ── Device ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── Model ───────────────────────────────────────────────
module = import_module(f"models.{model_config['architecture']}")
Model = getattr(module, "Model")
model = Model(model_config).to(device)
nb_params = sum(p.view(-1).size()[0] for p in model.parameters())
print(f"Model: {nb_params:,} params")

# ── Full dataset sizes ──────────────────────────────────
trn_list_path = protocol_dir / "CFAD.cm.train.trl.txt"
d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
print(f"\nFull training set: {len(file_train)} files, {len(file_train)//batch_size} batches (bs={batch_size})")

dev_trial_path = protocol_dir / "CFAD.cm.dev.trl.txt"
_, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
print(f"Dev set: {len(file_dev)} files, {len(file_dev)//batch_size} batches")

eval_trial_path = protocol_dir / "CFAD.cm.unseen.eval.trl.txt"
file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
print(f"Eval set: {len(file_eval)} files, {len(file_eval)//batch_size} batches")

# ════════════════════════════════════════════════════════
# BENCHMARK 1: Data loading speed (I/O bound)
# ════════════════════════════════════════════════════════
print("\n── Benchmark 1: Data loading ──")
train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=database_path / "train")
trn_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Time 50 batches of data loading only (no model)
t0 = time.time()
for i, (batch_x, batch_y) in enumerate(trn_loader):
    if i >= 50:
        break
data_time = time.time() - t0
print(f"  50 batches data loading: {data_time:.1f}s → {data_time/50:.2f}s/batch")

# ════════════════════════════════════════════════════════
# BENCHMARK 2: Training step (forward + backward + optim)
# ════════════════════════════════════════════════════════
print("\n── Benchmark 2: Training step (fwd+bwd+opt) ──")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

# Fresh loader for consistent measurement
trn_loader2 = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

model.train()
# Warmup (MPS needs warmup)
for i, (batch_x, batch_y) in enumerate(trn_loader2):
    if i >= 3:
        break
    batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
    _, _ = model(batch_x)

# Timed run
t0 = time.time()
for i, (batch_x, batch_y) in enumerate(trn_loader2):
    if i >= 20:
        break
    batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
    _, batch_out = model(batch_x)
    loss = criterion(batch_out, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
train_time = time.time() - t0
print(f"  20 training steps: {train_time:.1f}s → {train_time/20:.2f}s/batch")

# ════════════════════════════════════════════════════════
# BENCHMARK 3: Evaluation (forward only)
# ════════════════════════════════════════════════════════
print("\n── Benchmark 3: Evaluation (forward only) ──")
dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev, base_dir=database_path / "dev")
dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, drop_last=False)

model.eval()
t0 = time.time()
batch_count = 0
with torch.no_grad():
    for batch_x, _ in dev_loader:
        batch_x = batch_x.to(device)
        _, _ = model(batch_x)
        batch_count += 1
        if batch_count >= 100:
            break
eval_time = time.time() - t0
print(f"  100 eval batches: {eval_time:.1f}s → {eval_time/100:.2f}s/batch")

# ════════════════════════════════════════════════════════
# ESTIMATES
# ════════════════════════════════════════════════════════
print("\n╔══════════════════════════════════════════╗")
print("║        EPOCH TIME ESTIMATES             ║")
print("╚══════════════════════════════════════════╝")

n_train_batches = len(file_train) // batch_size
n_dev_batches = len(file_dev) // batch_size
n_eval_batches = len(file_eval) // batch_size

sec_per_train_batch = train_time / 20
sec_per_eval_batch = eval_time / 100

train_epoch_sec = n_train_batches * sec_per_train_batch
dev_eval_sec = n_dev_batches * sec_per_eval_batch
eval_eval_sec = n_eval_batches * sec_per_eval_batch

# Total for 1 epoch (train + dev eval every epoch + maybe eval on best)
total_sec = train_epoch_sec + dev_eval_sec + eval_eval_sec  # assumes eval_all_best triggers once

def fmt(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}min"

print(f"  Training only:         {fmt(train_epoch_sec)} ({n_train_batches} batches × {sec_per_train_batch:.2f}s)")
print(f"  Dev evaluation:        {fmt(dev_eval_sec)} ({n_dev_batches} batches × {sec_per_eval_batch:.2f}s)")
print(f"  Eval evaluation:       {fmt(eval_eval_sec)} ({n_eval_batches} batches × {sec_per_eval_batch:.2f}s)")
print(f"  ─────────────────────────────────────")
print(f"  Total 1 epoch:         ≈ {fmt(total_sec)}")
print()
print(f"  Note: eval_all_best=True, evaluation runs whenever best dev model is found.")
print(f"  In early epochs this likely triggers at least once.")
