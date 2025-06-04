import json
import os
import gc

import click
import torch
import numpy as np
from diffusers import AutoencoderTiny

import src.sdxl_custom_pipeline as sdxl_pipeline
from src.custom_dca import patch_pipe, reset_patched_unet


@click.command()
@click.option("--config_dir", nargs=1, type=str, default=None)
@click.option("--model_name", nargs=1, type=click.Choice(["hyper", "lightning", "lcm", "turbo"]), default="hyper", help="distillation checkpoint to use")
@click.option("--ips", nargs=1, type=str, default="0.8", help="list of ip_adapter_scale values")
@click.option("--lora_scale", nargs=1, type=float, default=0.6, help="lora_scale for additional loras inside unet, correspond to 'human-centric' bias")
@click.option("--id_img_path", nargs=1, type=str, default=None, help="path to image with identity")
@click.option("--prompt", nargs=1, type=str, default=None, help="text description for generation")
@click.option("--device", nargs=1, type=str, default="cuda:0")
@click.option("--seed", nargs=1, type=int, default=42, help="random seed for generation")
def main(
    config_dir, 
    model_name,
    ips, 
    lora_scale,
    id_img_path,
    prompt, 
    device, 
    seed
):
    assert id_img_path is not None, "id_img_path should be provided"
    assert prompt is not None, "prompt should be provided"
    
    DEVICE=torch.device(device)    
    with open(os.path.join(config_dir), "r") as f:
        conf = json.load(f)

    conf["model_name"] = model_name
    conf["pipe_kwargs"]["num_inference_steps"] = 4
    default_hw = (1024, 1024)
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
    pipe = sdxl_pipeline.name2pipe["faceid"](conf, DEVICE)
    
    # for colab demo use pruned cheap vae
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", 
        torch_dtype=torch.float16,
        cache_dir="models_cache/"
    ).to(DEVICE)
    torch.cuda.empty_cache()
    gc.collect()

    if conf["method"] == "faceid" and "faceid_lora_scale" in conf:
        assert "faceid_0" in pipe.unet.peft_config
        pipe.set_adapters(["faceid_0"], conf["faceid_lora_scale"])

    if conf["method"] == "faceid" and lora_scale is not None: # override with passed option
        pipe.set_adapters(["faceid_0"], lora_scale)

    if conf["method"] == "faceid" and ips is not None:
        pipe.set_ip_adapter_scale(ips)
        
    patch_pipe(pipe, **conf["am_patch_kwargs"]) if conf["patch_pipe"] else None
        
    ips_values = [float(v) for v in ips.split(" ")]
    assert np.all(np.array(ips_values) > 0), "ip_adapter_scale values should be positive"
    
    os.makedirs("output", exist_ok=True)
    for ipsv in ips_values:
        pipe.set_ip_adapter_scale(ipsv)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        with torch.no_grad():                
            img = pipe.execute(
                id_img_path,
                prompt,
                generator,
                conf["pipe_kwargs"],
                after_hook_fn=reset_patched_unet if conf["patch_pipe"] else lambda *args, **kwargs: None,
            )
        img.save(f"output/example_{ipsv}.png")

if __name__ == "__main__":
    main()