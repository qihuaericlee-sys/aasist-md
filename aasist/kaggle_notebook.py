# ╔══════════════════════════════════════════════════════════╗
# ║  AASIST × CFAD Mandarin — Kaggle Training               ║
# ║  Dataset: cfad-aasist-dataset                           ║
# ║  Config:  batch_size=16, epochs=100, GPU T4 x2          ║
# ║  Feature: 断点续训 (auto resume from checkpoint)         ║
# ╚══════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════
# CELL 1 — 挂载数据集 & 安装依赖
# ═══════════════════════════════════════════════════════════
#
# 1. 右侧面板 → Input → Add Data → 搜索 "cfad-aasist-dataset" → 添加
# 2. 右侧面板 → Settings → Accelerator → GPU T4 x2
#
# 如果这是续训: 确保之前的 checkpoint.pth 已上传到 dataset 的 code/ 目录下
#
# 然后运行此 Cell:

import os

print("=== Available datasets ===")
for d in sorted(os.listdir("/kaggle/input")):
    print(f"  /kaggle/input/{d}/")
    for f in sorted(os.listdir(f"/kaggle/input/{d}")):
        full = f"/kaggle/input/{d}/{f}"
        if os.path.isdir(full):
            try:
                count = len(os.listdir(full))
                print(f"    {f}/  ({count} items)")
            except:
                print(f"    {f}/")
        else:
            print(f"    {f}")

# 安装依赖
!pip install -q soundfile torchcontrib

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")
        

lines = open("/kaggle/working/code/main.py").readlines()
out = []
for i, line in enumerate(lines):
    out.append(line)
    if line.strip() == "optim.step()":
        sp = line[:len(line)-len(line.lstrip())]
        out.append(sp + 'if ii % 200 == 0:\n')
        out.append(sp + '    print("  batch " + str(ii) + "/" + str(total_batches) + "  loss=" + str(round(running_loss/num_total, 4)))\n')
open("/kaggle/working/code/main.py", "w").writelines(out)
print("done")

# ═══════════════════════════════════════════════════════════
# CELL 2 — 配置路径 & 导入代码
# ═══════════════════════════════════════════════════════════

import sys, shutil
from pathlib import Path

DATASET    = "datasets/qihuaericlee/cfad-aasist-dataset"
CODE_DIR   = Path(f"/kaggle/input/{DATASET}/code")
DB_PATH    = Path(f"/kaggle/input/{DATASET}/Mandarin_flac")
PROTO_DIR  = Path(f"/kaggle/input/{DATASET}/CFAD_cm_protocols_rock/CFAD_cm_protocols")
OUTPUT_DIR = Path("/kaggle/working/exp_result")

# 复制代码到可写目录
WORKING_CODE = Path("/kaggle/working/code")
if not WORKING_CODE.exists():
    shutil.copytree(CODE_DIR, WORKING_CODE)
sys.path.insert(0, str(WORKING_CODE))
CONFIG = WORKING_CODE / "config" / "AASIST_kaggle.conf"

# 验证路径
print("=== Path Check ===")
checks = [
    ("Code dir",      WORKING_CODE),
    ("Config file",   CONFIG),
    ("Train audio",   DB_PATH / "train" / "flac"),
    ("Dev audio",     DB_PATH / "dev" / "flac"),
    ("Eval unseen",   DB_PATH / "eval_unseen" / "flac"),
    ("Train proto",   PROTO_DIR / "CFAD.cm.train.trl.txt"),
    ("Dev proto",     PROTO_DIR / "CFAD.cm.dev.trl.txt"),
    ("Unseen proto",  PROTO_DIR / "CFAD.cm.unseen.eval.trl.txt"),
]
all_ok = True
for name, p in checks:
    ok = p.exists()
    status = "✓" if ok else "✗ MISSING"
    if not ok: all_ok = False
    print(f"  {status}: {name}")
if not all_ok:
    raise FileNotFoundError("Some paths missing — check your dataset structure!")
print("✓ All paths verified!\n")

# 检查是否有 checkpoint
ckpt = OUTPUT_DIR / "LA_AASIST_ep100_bs16_kaggle_100ep" / "weights" / "checkpoint.pth"
if ckpt.exists():
    print(f"✅ Found checkpoint → will RESUME from epoch {torch.load(ckpt, map_location='cpu', weights_only=True)['epoch']+1}")
else:
    print("🆕 No checkpoint found → training from scratch")

# ═══════════════════════════════════════════════════════════
# CELL 3 — 启动训练
# ═══════════════════════════════════════════════════════════

import json
from main import main
import argparse

with open(CONFIG) as f:
    cfg = json.load(f)
print(f"Config: {cfg['num_epochs']} epochs, batch_size={cfg['batch_size']}")

args = argparse.Namespace(
    config=str(CONFIG),
    output_dir=str(OUTPUT_DIR),
    seed=1234,
    eval=False,
    comment="kaggle_100ep",
    eval_model_weights=None,
    database_path=str(DB_PATH),
    protocol_dir=str(PROTO_DIR),
    eval_split="unseen",
    resume=True,   # ← 断点续训: 如果存在 checkpoint 则自动恢复
)

print("\n" + "=" * 60)
print("STARTING TRAINING (with auto-resume)")
print("=" * 60)

main(args)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# CELL 4 — 查看结果 & 打包下载
# ═══════════════════════════════════════════════════════════

import shutil, glob

exp_dirs = sorted(OUTPUT_DIR.glob("LA_AASIST_*"))
if exp_dirs:
    latest = exp_dirs[-1]
    print(f"=== Result: {latest.name} ===\n")

    # 训练日志
    metric_file = latest / "metric_log.txt"
    if metric_file.exists():
        print("--- Training Log ---")
        print(metric_file.read_text())

    # 最终指标
    tdcf_file = latest / "t-DCF_EER.txt"
    if tdcf_file.exists():
        print("--- Final t-DCF / EER ---")
        print(tdcf_file.read_text())

    # 列出保存的模型
    weights_dir = latest / "weights"
    if weights_dir.exists():
        print("--- Saved Models ---")
        for f in sorted(weights_dir.glob("*.pth")):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name} ({size_mb:.1f} MB)")

# 打包所有结果
archive_path = "/kaggle/working/aasist_results"
shutil.make_archive(archive_path, "zip", str(OUTPUT_DIR))
print(f"\n✓ Results archive: {archive_path}.zip")
print("  → 在 Kaggle 右侧 Output 面板下载")

# ═══════════════════════════════════════════════════════════
# CELL 5 — 持久化 checkpoint (用于续训)
# ═══════════════════════════════════════════════════════════
#
# 如果训练被中断或需要续训:
#   1. 下载上面的 aasist_results.zip
#   2. 解压后找到 checkpoint.pth
#   3. 上传到 cfad-aasist-dataset 的 code/ 目录下（New Version）
#   4. 重新运行本 Notebook → 自动从 checkpoint 续训
#
# 或者通过 Kaggle API 上传 checkpoint 到 dataset:
#   !pip install -q kaggle
#   # 需要先设置 kaggle.json credentials
#   !kaggle datasets version cfad-aasist-dataset \
#        -p /kaggle/working/... \
#        -m "Add checkpoint epoch XX"

print("Done! Download aasist_results.zip from the Output panel.")
