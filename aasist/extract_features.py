"""
Feature extraction script for AASIST pretrained model.

Extracts intermediate representations from the first K residual blocks
of the AASIST encoder as frame-level acoustic features.

The AASIST encoder consists of 6 cascaded Residual Blocks operating on
a 2D spectro-temporal representation derived from a learnable SincConv
filterbank. Each block applies Conv2d -> BN -> SELU -> residual addition
-> MaxPool along the temporal axis, progressively increasing channel
dimensionality while reducing temporal resolution.

By default, this script extracts the output of the first 3 blocks
(encoder[0..2]), yielding feature maps of shape:
    Block 0: (batch, 32, 23, 7163)
    Block 1: (batch, 32, 23, 2387)
    Block 2: (batch, 64, 23, 795)
for the default 64600-sample input (~4s at 16 kHz).

These intermediate features preserve rich spectro-temporal information
before the higher layers specialise for the anti-spoofing objective,
making them suitable for downstream transfer learning tasks.

Two output modes are supported:
    - "per_utt":  one .npy file per utterance per block (flexible, large
                  number of files).
    - "concat":   one large .npy file per block containing all utterances
                  stacked along axis 0 (fast I/O, easy to load).

Usage:
    python extract_features.py \
        --config config/AASIST.conf \
        --database_path ../data/LA \
        --model_path models/weights/AASIST.pth \
        --output_dir ./features \
        --subset eval \
        --num_blocks 3 \
        --output_mode concat

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-
    Temporal Graph Attention Networks," ICASSP 2022.
"""

import argparse
import json
import os
import sys
import time
import warnings
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Feature extractor wrapper
# ---------------------------------------------------------------------------

class AASISTFeatureExtractor(nn.Module):
    """Wrapper around the AASIST encoder for intermediate feature extraction.

    Given a pretrained AASIST model, this module re-uses its front-end
    (SincConv + BN + activation) and the first *num_blocks* residual
    blocks of the encoder.  The forward pass returns a list of feature
    maps, one per extracted block.

    Parameters
    ----------
    model : nn.Module
        A fully-initialised AASIST Model instance (weights loaded).
    num_blocks : int
        Number of leading encoder blocks whose outputs are collected.
        Must be in [1, 6].  Default is 3.
    """

    def __init__(self, model: nn.Module, num_blocks: int = 3):
        super().__init__()
        assert 1 <= num_blocks <= 6, "num_blocks must be in [1, 6]"
        self.num_blocks = num_blocks

        self.conv_time = model.conv_time
        self.first_bn = model.first_bn
        self.selu = nn.SELU(inplace=False)
        self.encoder_blocks = nn.ModuleList(
            [model.encoder[i][0] for i in range(num_blocks)]
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate feature maps.

        Parameters
        ----------
        x : Tensor, shape (batch, samples)
            Raw waveform input (16 kHz, mono).

        Returns
        -------
        features : list[Tensor]
            features[i] is the output of the i-th residual block,
            shape (batch, C_i, F_i, T_i).
        """
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=False)
        x = x.unsqueeze(dim=1)
        x = torch.nn.functional.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        features: List[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)

        return features


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str, model_path: str, device: str) -> nn.Module:
    """Instantiate AASIST and load pretrained weights."""
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    model_config = config["model_config"]

    module = import_module("models.{}".format(model_config["architecture"]))
    model = getattr(module, "Model")(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded from {model_path}  (#params: {nb_params:,})")
    return model


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def build_dataloader(database_path: Path, track: str, subset: str,
                     batch_size: int) -> Tuple[DataLoader, list]:
    """Build a DataLoader for the requested subset."""
    prefix = f"ASVspoof2019.{track}"

    if subset == "train":
        meta_path = (database_path /
                     f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.train.trn.txt")
        audio_dir = database_path / f"ASVspoof2019_{track}_train/"
        d_label, file_list = genSpoof_list(meta_path, is_train=True)
        dataset = Dataset_ASVspoof2019_train(
            list_IDs=file_list, labels=d_label, base_dir=audio_dir)
    elif subset == "dev":
        meta_path = (database_path /
                     f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.dev.trl.txt")
        audio_dir = database_path / f"ASVspoof2019_{track}_dev/"
        _, file_list = genSpoof_list(meta_path, is_train=False, is_eval=False)
        dataset = Dataset_ASVspoof2019_devNeval(
            list_IDs=file_list, base_dir=audio_dir)
    elif subset == "eval":
        meta_path = (database_path /
                     f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.eval.trl.txt")
        audio_dir = database_path / f"ASVspoof2019_{track}_eval/"
        file_list = genSpoof_list(meta_path, is_train=False, is_eval=True)
        dataset = Dataset_ASVspoof2019_devNeval(
            list_IDs=file_list, base_dir=audio_dir)
    else:
        raise ValueError(f"Unknown subset: {subset}")

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False,
                        pin_memory=False, num_workers=0)
    return loader, file_list


# ---------------------------------------------------------------------------
# Extraction: concatenated mode (one big file per block)
# ---------------------------------------------------------------------------

def extract_concat(extractor: AASISTFeatureExtractor,
                   loader: DataLoader,
                   utt_ids: list,
                   device: str,
                   output_dir: Path,
                   num_blocks: int,
                   subset: str) -> None:
    """Extract features and save one .npy per block (all utts stacked)."""
    extractor.eval()
    total_batches = len(loader)

    # Accumulate per-block results in lists, then stack at the end
    accum: List[List[np.ndarray]] = [[] for _ in range(num_blocks)]
    collected_ids: List[str] = []

    print(f"[INFO] Extracting {num_blocks}-block features for "
          f"'{subset}' ({len(loader.dataset)} utterances) …")
    t0 = time.time()
    utt_idx = 0

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            wav, keys = batch
        else:
            wav = batch
            keys = None

        wav = wav.to(device)
        features = extractor(wav)

        bs = wav.size(0)
        for i in range(bs):
            uid = utt_ids[utt_idx]
            collected_ids.append(uid)
            for b in range(num_blocks):
                accum[b].append(features[b][i].cpu().numpy())
            utt_idx += 1

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            speed = utt_idx / elapsed if elapsed > 0 else 0
            print(f"  [{batch_idx+1:>5}/{total_batches}]  "
                  f"{utt_idx:>6} utts  |  {speed:.1f} utt/s", flush=True)

    elapsed = time.time() - t0
    print(f"[INFO] Inference done: {utt_idx} utterances in {elapsed:.1f}s "
          f"({utt_idx/elapsed:.1f} utt/s)")

    # ---- Save ----
    print("[INFO] Saving features …")
    os.makedirs(output_dir, exist_ok=True)

    # Save utterance ID list
    id_path = output_dir / "utt_ids.json"
    with open(id_path, "w") as f:
        json.dump(collected_ids, f)

    # Save per-block arrays
    for b in range(num_blocks):
        arr = np.stack(accum[b], axis=0)  # (N, C, F, T)
        save_path = output_dir / f"block_{b}.npy"
        np.save(save_path, arr)
        print(f"  block_{b}: shape={arr.shape}, "
              f"size={arr.nbytes / 1e9:.2f} GB -> {save_path}")

    # Metadata
    meta = {
        "model": "AASIST",
        "subset": subset,
        "num_blocks": num_blocks,
        "num_utterances": utt_idx,
        "feature_shapes": {
            f"block_{b}": list(np.stack(accum[b][:1]).shape[1:])
            for b in range(num_blocks)
        },
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] All saved to {output_dir}")


# ---------------------------------------------------------------------------
# Extraction: per-utterance mode
# ---------------------------------------------------------------------------

def extract_per_utt(extractor: AASISTFeatureExtractor,
                    loader: DataLoader,
                    utt_ids: list,
                    device: str,
                    output_dir: Path,
                    num_blocks: int,
                    subset: str) -> None:
    """Extract features and save one .npy per utterance per block."""
    extractor.eval()
    total_batches = len(loader)

    block_dirs = []
    for b in range(num_blocks):
        d = output_dir / f"block_{b}"
        os.makedirs(d, exist_ok=True)
        block_dirs.append(d)

    print(f"[INFO] Extracting {num_blocks}-block features for "
          f"'{subset}' ({len(loader.dataset)} utterances) …")
    t0 = time.time()
    utt_idx = 0

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            wav, keys = batch
        else:
            wav = batch
            keys = None

        wav = wav.to(device)
        features = extractor(wav)

        bs = wav.size(0)
        for i in range(bs):
            uid = utt_ids[utt_idx]
            for b, feat in enumerate(features):
                np.save(block_dirs[b] / f"{uid}.npy", feat[i].cpu().numpy())
            utt_idx += 1

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            speed = utt_idx / elapsed if elapsed > 0 else 0
            print(f"  [{batch_idx+1:>5}/{total_batches}]  "
                  f"{utt_idx:>6} utts  |  {speed:.1f} utt/s", flush=True)

    elapsed = time.time() - t0
    print(f"[INFO] Extraction complete: {utt_idx} utterances in "
          f"{elapsed:.1f}s ({utt_idx/elapsed:.1f} utt/s)")

    meta = {
        "model": "AASIST",
        "subset": subset,
        "num_blocks": num_blocks,
        "num_utterances": utt_idx,
        "output_mode": "per_utt",
        "feature_shapes": {
            f"block_{b}": list(features[b].shape[1:])
            for b in range(num_blocks)
        },
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] All saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract intermediate residual-block features from a "
                    "pretrained AASIST model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", type=str, required=True,
                        help="Path to AASIST configuration file (.conf)")
    parser.add_argument("--database_path", type=str, required=True,
                        help="Root directory of the ASVspoof2019 LA dataset")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="./features",
                        help="Directory to store extracted features")
    parser.add_argument("--subset", type=str, default="eval",
                        choices=["train", "dev", "eval"],
                        help="Dataset subset to process")
    parser.add_argument("--num_blocks", type=int, default=3,
                        choices=range(1, 7),
                        help="Number of leading encoder blocks to extract")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Inference batch size")
    parser.add_argument("--output_mode", type=str, default="concat",
                        choices=["concat", "per_utt"],
                        help="'concat': one .npy per block (all utts stacked);"
                             " 'per_utt': one .npy per utterance per block")

    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Device: {device}")

    with open(args.config, "r") as f:
        config = json.loads(f.read())
    track = config["track"]

    database_path = Path(args.database_path)
    output_dir = Path(args.output_dir) / args.subset
    os.makedirs(output_dir, exist_ok=True)

    model_path = args.model_path or config["model_path"]
    model = load_model(args.config, model_path, device)

    extractor = AASISTFeatureExtractor(model, num_blocks=args.num_blocks)
    extractor = extractor.to(device).eval()

    loader, utt_ids = build_dataloader(
        database_path, track, args.subset, args.batch_size)
    print(f"[INFO] {args.subset} set: {len(loader.dataset)} utterances, "
          f"{len(loader)} batches")

    if args.output_mode == "concat":
        extract_concat(extractor, loader, utt_ids, device,
                       output_dir, args.num_blocks, args.subset)
    else:
        extract_per_utt(extractor, loader, utt_ids, device,
                        output_dir, args.num_blocks, args.subset)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
