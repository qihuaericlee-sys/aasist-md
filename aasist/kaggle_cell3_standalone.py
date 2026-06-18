# ╔══════════════════════════════════════════════════════════════╗
# ║  Cell 3 — COMPLETE STANDALONE TRAINING (no file imports)    ║
# ║  Copy this entire block into a single Kaggle cell           ║
# ╚══════════════════════════════════════════════════════════════╝

import sys, os, time, json, warnings, random
from pathlib import Path
from shutil import copy
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# 1. Load modules via importlib.util (bypass broken import)
# ═══════════════════════════════════════════════════════════════

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CODE = Path("/kaggle/working/code")

# Load data_utils
du = load_module("data_utils", str(CODE / "data_utils.py"))
genSpoof_list = du.genSpoof_list
Dataset_ASVspoof2019_train = du.Dataset_ASVspoof2019_train
Dataset_ASVspoof2019_devNeval = du.Dataset_ASVspoof2019_devNeval

# Load evaluation
ev = load_module("evaluation", str(CODE / "evaluation.py"))
calculate_EER = ev.calculate_EER

# Load model
models_mod = load_module("models.AASIST", str(CODE / "models" / "AASIST.py"))
Model = models_mod.Model

# ═══════════════════════════════════════════════════════════════
# 2. Utility functions (inlined from utils.py)
# ═══════════════════════════════════════════════════════════════

def str_to_bool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'): return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'): return False
    raise ValueError(f'invalid truth value {val}')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def create_optimizer(model_parameters, optim_config):
    optimizer_name = optim_config['optimizer']
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(optim_config['amsgrad']))
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    if optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step, total_steps, 1, optim_config['lr_min'] / optim_config['base_lr']))
    else:
        scheduler = None
    return optimizer, scheduler

# ═══════════════════════════════════════════════════════════════
# 3. Paths & config
# ═══════════════════════════════════════════════════════════════

DATASET     = "datasets/qihuaericlee/cfad-aasist-dataset"
DB_PATH     = Path(f"/kaggle/input/{DATASET}/Mandarin_flac")
PROTO_DIR   = Path(f"/kaggle/input/{DATASET}/CFAD_cm_protocols_rock/CFAD_cm_protocols")
OUTPUT_DIR  = Path("/kaggle/working/exp_result")
CONFIG_PATH = CODE / "config" / "AASIST_kaggle.conf"
EVAL_SPLIT  = "unseen"
SEED        = 1234

with open(CONFIG_PATH) as f:
    config = json.load(f)

print(f"Config: {config['num_epochs']} epochs, bs={config['batch_size']}")

model_config = config["model_config"]
optim_config = config["optim_config"]
optim_config["epochs"] = config["num_epochs"]

# ═══════════════════════════════════════════════════════════════
# 4. Device & model
# ═══════════════════════════════════════════════════════════════

set_seed(SEED, config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = Model(model_config).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ═══════════════════════════════════════════════════════════════
# 5. DataLoaders
# ═══════════════════════════════════════════════════════════════

# Train
d_label_trn, file_train = genSpoof_list(
    dir_meta=PROTO_DIR / "CFAD.cm.train.trl.txt", is_train=True, is_eval=False)
print(f"Training files: {len(file_train)}")

train_set = Dataset_ASVspoof2019_train(
    list_IDs=file_train, labels=d_label_trn, base_dir=DB_PATH / "train")
gen = torch.Generator()
gen.manual_seed(SEED)
trn_loader = DataLoader(train_set, batch_size=config["batch_size"],
                         shuffle=True, drop_last=True, pin_memory=True,
                         worker_init_fn=seed_worker, generator=gen)

# Dev
_, file_dev = genSpoof_list(
    dir_meta=PROTO_DIR / "CFAD.cm.dev.trl.txt", is_train=False, is_eval=False)
print(f"Dev files: {len(file_dev)}")

dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev, base_dir=DB_PATH / "dev")
dev_loader = DataLoader(dev_set, batch_size=config["batch_size"],
                         shuffle=False, drop_last=False, pin_memory=True)

# Eval
file_eval = genSpoof_list(
    dir_meta=PROTO_DIR / f"CFAD.cm.{EVAL_SPLIT}.eval.trl.txt",
    is_train=False, is_eval=True)
eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, base_dir=DB_PATH / f"eval_{EVAL_SPLIT}")
eval_loader = DataLoader(eval_set, batch_size=config["batch_size"],
                          shuffle=False, drop_last=False, pin_memory=True)

dev_trial_path = PROTO_DIR / "CFAD.cm.dev.trl.txt"
eval_trial_path = PROTO_DIR / f"CFAD.cm.{EVAL_SPLIT}.eval.trl.txt"

# ═══════════════════════════════════════════════════════════════
# 6. Setup output dirs
# ═══════════════════════════════════════════════════════════════

model_tag = OUTPUT_DIR / f"LA_AASIST_ep{config['num_epochs']}_bs{config['batch_size']}_kaggle_100ep"
model_save_path = model_tag / "weights"
metric_path = model_tag / "metrics"
eval_score_path = model_tag / config["eval_output"]
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(metric_path, exist_ok=True)
copy(CONFIG_PATH, model_tag / "config.conf")

# ═══════════════════════════════════════════════════════════════
# 7. Optimizer & resume
# ═══════════════════════════════════════════════════════════════

optim_config["steps_per_epoch"] = len(trn_loader)
optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
optimizer_swa = SWA(optimizer)

best_dev_eer = 1.0
best_eval_eer = 100.0
best_dev_tdcf = 0.05
best_eval_tdcf = 1.0
n_swa_update = 0
start_epoch = 0
checkpoint_path = model_save_path / "checkpoint.pth"

if checkpoint_path.exists():
    print(f"Resuming from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if ckpt.get("scheduler_state") and scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_dev_eer = ckpt.get("best_dev_eer", 1.0)
    best_eval_eer = ckpt.get("best_eval_eer", 100.0)
    best_dev_tdcf = ckpt.get("best_dev_tdcf", 0.05)
    best_eval_tdcf = ckpt.get("best_eval_tdcf", 1.0)
    n_swa_update = ckpt.get("n_swa_update", 0)
    print(f"Resumed epoch {start_epoch}, dev_eer={best_dev_eer:.4f}")
else:
    print("Fresh training from scratch")

f_log = open(model_tag / "metric_log.txt", "a")
f_log.write("=====\n")
writer = SummaryWriter(model_tag)

# ═══════════════════════════════════════════════════════════════
# 8. Training loop
# ═══════════════════════════════════════════════════════════════

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

for epoch in range(start_epoch, config["num_epochs"]):
    t0 = time.time()
    print(f"\n{'='*50}\nStart training epoch {epoch:03d}\n{'='*50}")

    # ── Train one epoch ──────────────────────────────────
    model.train()
    running_loss = 0.0
    num_total = 0.0
    total_batches = len(trn_loader)

    for ii, (batch_x, batch_y) in enumerate(trn_loader):
        bs = batch_x.size(0)
        num_total += bs
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x)
        loss = criterion(batch_out, batch_y)
        running_loss += loss.item() * bs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if (ii + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ii + 1) * (total_batches - ii - 1)
            print(f"  batch {ii+1:5d}/{total_batches}  "
                  f"loss={running_loss/num_total:.4f}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    running_loss /= num_total
    print(f"Epoch {epoch:03d} training done — loss={running_loss:.5f}  "
          f"time={(time.time()-t0)/60:.1f}min")

    # ── Dev evaluation ──────────────────────────────────
    model.eval()
    dev_scores = []
    dev_keys = []
    with open(dev_trial_path) as f:
        dev_lines = f.readlines()
    for batch_x, utt_id in dev_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = batch_out[:, 1].cpu().numpy().ravel()
        dev_keys.extend(utt_id)
        dev_scores.extend(batch_score.tolist())

    dev_score_file = metric_path / "dev_score.txt"
    with open(dev_score_file, "w") as fh:
        for fn, sco, trl in zip(dev_keys, dev_scores, dev_lines):
            _, uid, _, src, key = trl.strip().split(' ')
            fh.write(f"{uid} {src} {key} {sco}\n")

    dev_eer, dev_tdcf = calculate_EER(
        cm_scores_file=dev_score_file,
        output_file=metric_path / f"dev_t-DCF_EER_{epoch}epo.txt",
        printout=False)
    print(f"DONE. Loss:{running_loss:.5f}, dev_eer:{dev_eer:.3f}, dev_tdcf:{dev_tdcf:.5f}")

    # ── Best model tracking ─────────────────────────────
    best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
    if best_dev_eer >= dev_eer:
        print(f"*** best model at epoch {epoch} ***")
        best_dev_eer = dev_eer
        torch.save(model.state_dict(),
                   model_save_path / f"epoch_{epoch}_{dev_eer:03.3f}.pth")

        # Eval on test set
        eval_scores_list = []
        eval_keys_list = []
        with open(eval_trial_path) as f:
            eval_lines = f.readlines()
        for batch_x, utt_id in eval_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = batch_out[:, 1].cpu().numpy().ravel()
            eval_keys_list.extend(utt_id)
            eval_scores_list.extend(batch_score.tolist())

        with open(eval_score_path, "w") as fh:
            for fn, sco, trl in zip(eval_keys_list, eval_scores_list, eval_lines):
                _, uid, _, src, key = trl.strip().split(' ')
                fh.write(f"{uid} {src} {key} {sco}\n")

        eval_eer, eval_tdcf = calculate_EER(
            cm_scores_file=eval_score_path,
            output_file=metric_path / f"t-DCF_EER_{epoch:03d}epo.txt")

        log_text = f"epoch{epoch:03d}, "
        if eval_eer < best_eval_eer:
            log_text += f"best eer, {eval_eer:.4f}%"
            best_eval_eer = eval_eer
        if eval_tdcf < best_eval_tdcf:
            log_text += f"best tdcf, {eval_tdcf:.4f}"
            best_eval_tdcf = eval_tdcf
            torch.save(model.state_dict(), model_save_path / "best.pth")
        if log_text:
            print(log_text)
            f_log.write(log_text + "\n")

        optimizer_swa.update_swa()
        n_swa_update += 1

    writer.add_scalar("loss", running_loss, epoch)
    writer.add_scalar("dev_eer", dev_eer, epoch)
    writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
    writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    # ── Save checkpoint ─────────────────────────────────
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_dev_eer": best_dev_eer,
        "best_eval_eer": best_eval_eer,
        "best_dev_tdcf": best_dev_tdcf,
        "best_eval_tdcf": best_eval_tdcf,
        "n_swa_update": n_swa_update,
    }, checkpoint_path)
    print(f"Checkpoint saved: epoch {epoch}")

# ═══════════════════════════════════════════════════════════════
# 9. Final evaluation
# ═══════════════════════════════════════════════════════════════

print("\nStart final evaluation...")
epoch += 1
if n_swa_update > 0:
    optimizer_swa.swap_swa_sgd()
    optimizer_swa.bn_update(trn_loader, model, device=device)

# Final eval
eval_scores_list = []
eval_keys_list = []
with open(eval_trial_path) as f:
    eval_lines = f.readlines()
for batch_x, utt_id in eval_loader:
    batch_x = batch_x.to(device)
    with torch.no_grad():
        _, batch_out = model(batch_x)
        batch_score = batch_out[:, 1].cpu().numpy().ravel()
    eval_keys_list.extend(utt_id)
    eval_scores_list.extend(batch_score.tolist())

with open(eval_score_path, "w") as fh:
    for fn, sco, trl in zip(eval_keys_list, eval_scores_list, eval_lines):
        _, uid, _, src, key = trl.strip().split(' ')
        fh.write(f"{uid} {src} {key} {sco}\n")

eval_eer, eval_tdcf = calculate_EER(
    cm_scores_file=eval_score_path,
    output_file=model_tag / "t-DCF_EER.txt")
f_log.write("=====\n")
f_log.write(f"EER: {eval_eer:.3f}, min t-DCF: {eval_tdcf:.5f}\n")
f_log.close()

torch.save(model.state_dict(), model_save_path / "swa.pth")
if eval_eer <= best_eval_eer:
    best_eval_eer = eval_eer
if eval_tdcf <= best_eval_tdcf:
    best_eval_tdcf = eval_tdcf
    torch.save(model.state_dict(), model_save_path / "best.pth")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Best EER: {best_eval_eer:.3f}%, min t-DCF: {best_eval_tdcf:.5f}")
print(f"{'='*60}")
