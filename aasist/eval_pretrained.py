"""
Evaluation script for AASIST pretrained model on ASVspoof2019 LA dataset.
Adapted for Apple Silicon (MPS) / CPU inference.
"""
import argparse
import json
import os
import sys
import warnings
import time
from pathlib import Path
from importlib import import_module
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from evaluation_fixed import calculate_tDCF_EER

warnings.filterwarnings("ignore", category=FutureWarning)


def get_device():
    """Select best available device: MPS > CPU"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params: {}".format(nb_params))
    return model


def produce_evaluation_file(data_loader, model, device, save_path, trial_path):
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []

    total_batches = len(data_loader)
    for i, (batch_x, utt_id) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        if (i + 1) % 100 == 0 or (i + 1) == total_batches:
            print(f"  Progress: {i+1}/{total_batches} batches", flush=True)

    assert len(trial_lines) == len(fname_list) == len(score_list), \
        f"Mismatch: trials={len(trial_lines)}, fnames={len(fname_list)}, scores={len(score_list)}"

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def main(args):
    # load config
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    track = config["track"]

    # paths
    database_path = Path(args.database_path)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019))

    eval_score_path = output_dir / "eval_scores.txt"
    result_file = output_dir / "t-DCF_EER.txt"

    # device
    device = get_device()
    print("Device: {}".format(device))

    # model
    model = get_model(model_config, device)
    model_path = args.model_path or config["model_path"]
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded: {}".format(model_path))

    # dataloader
    file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
    print("no. eval files: {}".format(len(file_eval)))
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False,
                             num_workers=0)

    # run evaluation
    print("Start evaluation...")
    start_time = time.time()
    produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
    elapsed = time.time() - start_time
    print("Inference completed in {:.1f} seconds".format(elapsed))

    # calculate metrics
    print("\nCalculating EER and t-DCF...")
    asv_score_file = database_path / config["asv_score_path"]
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=asv_score_file,
        output_file=result_file)

    # print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print("EER:       {:.4f} %".format(eval_eer))
    print("min t-DCF: {:.6f}".format(eval_tdcf))
    print("=" * 50)
    print("\nPaper reported: EER=0.83%, min t-DCF=0.0275")
    print("Detailed results saved to: {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST pretrained model evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="configuration file path")
    parser.add_argument("--database_path", type=str, required=True,
                        help="path to LA dataset directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="path to pretrained model weights (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./eval_result",
                        help="output directory for results")
    main(parser.parse_args())
