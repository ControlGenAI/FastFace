import json
from pathlib import Path
import os
import subprocess


import click
import numpy as np
import torch
from tqdm.auto import tqdm

from src.dataset import FaceIdDataset
from src.evaluation.metrics import ExperimentDataset
from src.evaluation.face_metrics import FaceDistanceMetric, FaceClipScore
from src.evaluation.clip_metric import CLIPMetric
from src.evaluation.aesthetic_metric import AestheticMetric
from src.evaluation.face_metrics import get_face_ratios


def process_raw_metrics(metrics_raw, exp_runs, metrics_knames, agg_metrics):
    """returns metrics dict averaged across experiments"""
    results = dict()
    for exp_name in tqdm(exp_runs):
        results[exp_name] = dict()
        for metric in metrics_knames:
            if metric in agg_metrics:
                results[exp_name][f"{metric}"] = metrics_raw[exp_name][metric]
                continue
            for k in metrics_raw[exp_name][metric].keys():
                values = metrics_raw[exp_name][metric][k]
                avm = np.mean(values)
                results[exp_name][f"{metric}/{k}"] = f"{avm:.3f}"
    return results


def metrics_flags_to_knames(
    include_facesim,
    include_clipscore,
    include_ir,
    include_faceclip,
):
    knames, agg_names = [], []
    if include_facesim:
        knames.append("face_d")
        knames.append("face_fail_cnt")
        agg_names.append("face_fail_cnt")
    if include_clipscore:
        knames.append("clip")
    if include_ir:
        knames.append("aesthetic")
    if include_faceclip:
        knames.append("face_clip")
    return knames, agg_names


@click.command()
@click.argument('exp_runs', nargs=-1, type=str, required=True,)
@click.option('--log_to', nargs=1, type=str, default=None)
@click.option('--data_dir', nargs=1, type=str, default="data/")
@click.option('--ds_promptset', nargs=1, type=click.Choice(["full", "realistic", "style"]), default="full")
@click.option('--full_subset', nargs=1, type=click.Choice([None, "realistic", "style"]), default=None, help="specify if experiment was computed on full data and only want to evaluate certain part") # None, realistic or style, use if experiment is full but need to evaluate only on part of propmts without recomputing rest
@click.option("--deviceid", nargs=1, type=int, default=0)
# metrics
@click.option("--include_facesim", is_flag=True, show_default=True, default=False, help="ID metric")
@click.option("--include_clipscore", is_flag=True, show_default=True, default=False, help="CLIP scores")
@click.option("--include_ir", is_flag=True, show_default=True, default=False, help="AE/IR metrics")
@click.option("--include_faceclip", is_flag=True, show_default=True, default=False, help="FCS metric")
# options
@click.option('--rthresh', nargs=1, type=float, default=None, help="face bb ratio threshold for evaluation")
@click.option("--save_raw", is_flag=True, show_default=True, default=False)
@click.option("--build_ratios", is_flag=True, show_default=True, default=False)
def main(
    exp_runs,
    log_to, 
    data_dir,
    ds_promptset,
    full_subset,
    deviceid,
    include_facesim,
    include_clipscore,
    include_ir,
    include_faceclip,
    rthresh, 
    save_raw,
    build_ratios,
):

    if not torch.cuda.is_available(): 
        raise ValueError("No GPU available")
    DEVICE = torch.device(f"cuda:{deviceid}")

    flagfail = 0
    for exp_dir in exp_runs:
        if not Path(exp_dir).exists():
            print(f"dir not found: {exp_dir}")
            flagfail += 1
    if flagfail > 0: 
        exit()

    def is_valid_example(img, prompt):
        if full_subset is None:
            return True
        elif full_subset == "realistic":
            return ";" not in prompt # ";" is used only in 'style' prompts
        elif full_subset == "style":
            return ";" in prompt
    
    dataset_kwargs = { # supposed to be same forall exps, TODO: attach them to exp dir with config
        "data_dir": data_dir,
        "prompts_set": ds_promptset,
    }

    print(f"ds kwargs: {str(dataset_kwargs)}")
    dataset = FaceIdDataset(**dataset_kwargs) # instantiate dataset for experiments
    print("dataset len: ", len(dataset))
    metrics_raw = {}

    metrics_knames, agg_metrics = metrics_flags_to_knames(
        include_facesim,
        include_clipscore,
        include_ir,
        include_faceclip,
    )

    print("target metrics: " + str(metrics_knames))

    if build_ratios: # make sure that ratios.json exists in each exp dir
        for exp_name in exp_runs:
            if os.path.exists(Path(exp_name) / "ratios.json"): 
                continue
            get_face_ratios(ExperimentDataset(exp_name, dataset, filter_subset=is_valid_example)) # file with ratios saved in exp dir

    # manually install these models into according paths
    if not os.path.exists('models_cache/ir'):
        print("Downloading Reward models...")
        os.makedirs('models_cache/ir', exist_ok=True)
        subprocess.run([
            "wget",
            "--content-disposition",
            "https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt",
            "-O", "models_cache/ir/ImageReward.pt"
        ])
        subprocess.run([
            "wget",
            "--content-disposition",
            "https://nxt.2a2i.org/index.php/s/xS7Jq7oqwiXNBad/download/ava+logos-l14-linearMSE.pth",
            "-O", "models_cache/ir/ava+logos-l14-linearMSE.pth"
        ])        
    ir_path = 'models_cache/ir/ImageReward.pt'
    acu_path = 'models_cache/ir/ava+logos-l14-linearMSE.pth'

    bad_exp_names = []
    for exp_name in tqdm(exp_runs):            
        print('[ CALCULATING ]', exp_name)
        metrics_raw[exp_name] = dict()

        try:
            if include_facesim:
                face_dist_metrics, face_fail_cnt = FaceDistanceMetric(
                    exp_name,
                    dataset,
                    device=DEVICE,
                    build_ratios=build_ratios,
                    filter_subset=is_valid_example,
                )(rthresh=rthresh)
        
                metrics_raw[exp_name]["face_d"] = face_dist_metrics
                metrics_raw[exp_name]["face_fail_cnt"] = face_fail_cnt # amount of times there was no face in image

            if include_clipscore:
                clip_scores = CLIPMetric(
                    exp_name,
                    dataset,
                    device=DEVICE,
                    build_ratios=build_ratios,
                    filter_subset=is_valid_example,
                )(rthresh=rthresh)
                
                metrics_raw[exp_name]["clip"] = clip_scores
            
            if include_ir:
                aesthetic_metrics = AestheticMetric(
                    exp_name,
                    dataset,
                    device=DEVICE,
                    ir_path=ir_path,
                    model_path=acu_path,
                    build_ratios=build_ratios,
                    filter_subset=is_valid_example,
                )(rthresh=rthresh)
                
                metrics_raw[exp_name]["aesthetic"] = aesthetic_metrics

            if include_faceclip:
                faceclip_scores, _ = FaceClipScore(
                    exp_name,
                    dataset,
                    device=DEVICE,
                    build_ratios=build_ratios,
                    filter_subset=is_valid_example,
                )(rthresh=rthresh)
                metrics_raw[exp_name]["face_clip"] = faceclip_scores 
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"Error in {exp_name}")
            bad_exp_names.append(exp_name)
            continue


    print("[EVALUATION RESULTS]\n\n")
    print(f"saved to: {log_to}")
    print(f"bad exp names (had exceptions during evaluations): {bad_exp_names}")
    results = process_raw_metrics(metrics_raw, exp_runs, metrics_knames, agg_metrics)

    print(json.dumps(results, indent=4))
    
    if log_to is not None:
        with open(f"{log_to}.json", "w") as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()