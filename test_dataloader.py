"""
快速数据加载测试脚本
目的: 在提交长时间训练前, 验证数据路径、协议文件和 DataLoader 都正常
用法: python test_dataloader.py
交互式: srun --gres=gpu:1 --cpus-per-task=4 --time=00:15:00 --pty python test_dataloader.py
"""
import sys
import os
import json
import time

# 确保能找到 aasist 子目录中的模块 (main.py, data_utils.py 所在)
CODE_DIR = "/home/comp/25450212/aasist/aasist"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import torch
import numpy as np
from pathlib import Path
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2019_devNeval
from torch.utils.data import DataLoader

CONFIG = {
    "database_path": "/home/comp/25450212/dataset/Mandarin_flac",
    "protocol_dir": "/home/comp/25450212/dataset/CFAD_cm_protocols_rock/CFAD_cm_protocols",
    "batch_size": 16,
    "track": "LA",
}

def test_split(name, db_subdir, protocol_file, is_train=False, is_eval=False):
    """测试单个数据分片"""
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")

    db_path = Path(CONFIG["database_path"]) / db_subdir
    protocol_path = Path(CONFIG["protocol_dir"]) / protocol_file

    # 1. 检查路径
    print(f"  Database dir:  {db_path}  -> exists: {db_path.exists()}")
    print(f"  Protocol file: {protocol_path}  -> exists: {protocol_path.exists()}")

    if not db_path.exists():
        print(f"  [FAIL] FAIL: Database directory not found!")
        return False
    if not protocol_path.exists():
        print(f"  [FAIL] FAIL: Protocol file not found!")
        return False

    # 2. 检查 flac 目录
    flac_dir = db_path / "flac"
    if not flac_dir.exists():
        print(f"  [FAIL] FAIL: flac/ subdirectory missing under {db_path}")
        return False
    n_files = len(list(flac_dir.glob("*.flac")))
    print(f"  .flac files in flac/: {n_files}")

    # 3. 解析协议
    if is_train:
        labels, file_list = genSpoof_list(str(protocol_path), is_train=True, is_eval=False)
        print(f"  Protocol entries: {len(file_list)}")
        # 统计标签
        n_spoof = sum(1 for v in labels.values() if v == 0)
        n_bona = sum(1 for v in labels.values() if v == 1)
        print(f"  Labels: {n_spoof} spoof / {n_bona} bonafide")
    elif is_eval:
        file_list = genSpoof_list(str(protocol_path), is_train=False, is_eval=True)
        print(f"  Protocol entries: {len(file_list)}")
    else:
        labels, file_list = genSpoof_list(str(protocol_path), is_train=False, is_eval=False)
        print(f"  Protocol entries: {len(file_list)}")
        n_spoof = sum(1 for v in labels.values() if v == 0)
        n_bona = sum(1 for v in labels.values() if v == 1)
        print(f"  Labels: {n_spoof} spoof / {n_bona} bonafide")

    # 4. 检查协议文件与实际 flac 文件的匹配率
    missing = 0
    for key in file_list[:200]:  # 抽查前200个
        fpath = flac_dir / f"{key}.flac"
        if not fpath.exists():
            missing += 1
    if missing > 0:
        print(f"  [WARN]  WARNING: {missing}/200 protocol entries have NO matching .flac file")
    else:
        print(f"  [PASS]  200/200 spot-check passed (protocol <-> files match)")

    # 5. 测试 Dataset 加载 (前5个样本)
    print(f"  Testing Dataset.__getitem__() ...")
    try:
        if is_train:
            ds = Dataset_ASVspoof2019_train(file_list[:5], labels, base_dir=db_path)
        else:
            ds = Dataset_ASVspoof2019_devNeval(file_list[:5], base_dir=db_path)
        for i in range(min(3, len(ds))):
            sample = ds[i]
            if is_train:
                audio, label = sample
                print(f"    [{i}] shape={audio.shape}, label={label}")
            else:
                audio, key = sample
                print(f"    [{i}] shape={audio.shape}, key={key}")
        print(f"  [PASS]  Dataset loading OK")
    except Exception as e:
        print(f"  [FAIL] FAIL: {e}")
        return False

    # 6. 测试 DataLoader
    print(f"  Testing DataLoader (batch_size={CONFIG['batch_size']}) ...")
    try:
        if is_train:
            ds = Dataset_ASVspoof2019_train(file_list, labels, base_dir=db_path)
            dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True,
                            drop_last=True, num_workers=2)
        else:
            ds = Dataset_ASVspoof2019_devNeval(file_list, base_dir=db_path)
            dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            drop_last=False, num_workers=2)

        t0 = time.time()
        for i, batch in enumerate(dl):
            if i >= 3:  # 只测3个batch
                break
            if is_train:
                audio, labels = batch
                print(f"    Batch {i}: audio={audio.shape}, labels={labels.shape}")
            else:
                audio, keys = batch
                print(f"    Batch {i}: audio={audio.shape}, keys={len(keys)}")
        t1 = time.time()
        print(f"  [PASS]  DataLoader OK ({t1 - t0:.2f}s for 3 batches)")
    except Exception as e:
        print(f"  [FAIL] FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("=" * 60)
    print("  AASIST Data Loading Test Suite")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print("=" * 60)

    results = []

    # Train
    results.append(test_split(
        "Train", "train", "CFAD.cm.train.trl.txt",
        is_train=True
    ))
    # Dev
    results.append(test_split(
        "Dev", "dev", "CFAD.cm.dev.trl.txt",
        is_train=False, is_eval=False
    ))
    # Eval seen
    results.append(test_split(
        "Eval (seen)", "eval_seen", "CFAD.cm.seen.eval.trl.txt",
        is_train=False, is_eval=True
    ))
    # Eval unseen
    results.append(test_split(
        "Eval (unseen)", "eval_unseen", "CFAD.cm.unseen.eval.trl.txt",
        is_train=False, is_eval=True
    ))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  [PASS]  ALL {total}/{total} TESTS PASSED -- ready to train!")
    else:
        print(f"  [FAIL]  {passed}/{total} passed, {total - passed} FAILED")
        print(f"  Fix the issues above before running full training.")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
