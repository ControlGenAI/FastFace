import json
import os
from pathlib import Path
import time

from tqdm import tqdm
import click
import torch

from src.dataset import FaceIdDataset
import src.sdxl_custom_pipeline as sdxl_pipeline
from src.custom_dca import patch_pipe, unpatch_unet, reset_patched_unet

SEED=50

def create_exp_root(out_dir, config_name, exp_name=None):
    ts = time.time()
    timestamp = int(round(ts))
    
    if exp_name is None:
        exp_name = f"{config_name}_{timestamp}"
    
    exp_root = os.path.join(out_dir, f"{exp_name}")
    if os.path.exists(exp_root):
        print('[ WARNING ] exp root exists, content will be overwritten')
    else:
        print('[ INFO ] exp root is created: ', str(exp_root))
    Path(exp_root).mkdir(parents=True, exist_ok=True)
    return exp_name, exp_root


@click.command()
@click.option("--target_adapter", nargs=1, type=click.Choice([
    "faceid", 
]), default="faceid")
@click.option("--config_dir", nargs=1, type=str, default=None)
@click.option("--data_dir", nargs=1, type=str, default="data/")
@click.option("--ds_type", nargs=1, type=click.Choice(["full", "realistic", "style"]), default="realistic")
@click.option("--ips", nargs=1, type=float, default=None)
@click.option("--lora_scale", nargs=1, type=float, default=None)
@click.option("--exp_title", nargs=1, type=str, default=None)
@click.option("--out_dir", nargs=1, type=str, default="res/")
@click.option("--device", nargs=1, type=str, default="cuda:1")
@click.option("--include_hyper_4", is_flag=True, show_default=True, default=False)
@click.option("--include_lightning_4", is_flag=True, show_default=True, default=False)
@click.option("--include_lcm_4", is_flag=True, show_default=True, default=False)
@click.option("--include_turbo_4", is_flag=True, show_default=True, default=False)
def main(
    target_adapter, 
    config_dir, 
    data_dir, 
    ds_type, 
    ips, 
    lora_scale, 
    exp_title, 
    out_dir, 
    device, 
    include_hyper_4, 
    include_lightning_4, 
    include_lcm_4, 
    include_turbo_4
):
    
    DEVICE=torch.device(device)
    if target_adapter == "pulid": # specific prompts for pulid evaluation
        ds_type += "_pure"
    eval_dataset = FaceIdDataset(data_dir, prompts_set=ds_type)

    target_models = []
    if include_hyper_4:
        target_models.append(("hyper", 4))
    if include_lightning_4:
        target_models.append(("lightning", 4))
    if include_lcm_4:
        target_models.append(("lcm", 4))
    if include_turbo_4:
        target_models.append(("turbo", 4))
    
    print("EVALUATING FOR: ", target_models)

    with open(os.path.join(config_dir), "r") as f:
        conf = json.load(f)
    
    conf_name = config_dir.split('/')[-1].split('.')[0]

    default_hw = conf["pipe_kwargs"]["height"], conf["pipe_kwargs"]["width"]
    for model_name, nsteps in target_models:
        conf["model_name"] = model_name
        conf["pipe_kwargs"]["num_inference_steps"] = nsteps

        if conf["reset_gs_to_default"]: # should do this only for configs where keep gs basic (like baseline or AM only)
            if model_name == "lcm":
                conf["pipe_kwargs"]["guidance_scale_a"] = 1.5
                conf["pipe_kwargs"]["guidance_scale_b"] = 1.5
            elif model_name == "base":
                conf["pipe_kwargs"]["guidance_scale_a"] = 5.
                conf["pipe_kwargs"]["guidance_scale_b"] = 5.
            else:
                conf["pipe_kwargs"]["guidance_scale_a"] = 1.
                conf["pipe_kwargs"]["guidance_scale_b"] = 1.
            
        if model_name == "turbo":
            conf["pipe_kwargs"]["height"] = 512
            conf["pipe_kwargs"]["width"] = 512
        else:
            conf["pipe_kwargs"]["height"] = default_hw[0]
            conf["pipe_kwargs"]["width"] = default_hw[1]

        pipe = sdxl_pipeline.name2pipe[target_adapter](conf, DEVICE)

        if conf["method"] == "faceid" and "faceid_lora_scale" in conf:
            assert "faceid_0" in pipe.unet.peft_config
            pipe.set_adapters(["faceid_0"], conf["faceid_lora_scale"])

        if conf["method"] == "faceid" and lora_scale is not None: # override with passed option
            pipe.set_adapters(["faceid_0"], lora_scale)

        if conf["method"] == "faceid" and ips is not None:
            pipe.set_ip_adapter_scale(ips)
        
        exp_name, exp_dir = create_exp_root(out_dir, conf_name, exp_title)
        print("Outputing to: ", exp_dir)

        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        if "patch_pipe" in conf and conf["patch_pipe"]:
            patch_pipe(pipe, **conf["am_patch_kwargs"])
        for idx, (cond_img, prompt) in enumerate(tqdm(eval_dataset)):
            with torch.no_grad():                
                img = pipe.execute(
                    os.path.join(eval_dataset.img_root, cond_img),
                    prompt,
                    generator,
                    conf["pipe_kwargs"],
                    after_hook_fn=reset_patched_unet if conf["patch_pipe"] else lambda _: None,
                )
            img.save(os.path.join(exp_dir, f"{idx}.png"))

        if conf["patch_pipe"]:
            unpatch_unet(pipe.unet)

        conf["dataset_meta"] = {
            "data_dir": data_dir,
            "promptset": ds_type,
        }

        with open(os.path.join(exp_dir, "config.json"), "w+") as f: # save config with hyper-params as well 
            json.dump(conf, f, indent=4)


if __name__ == "__main__":
    main()