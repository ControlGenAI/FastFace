import os
import json
import random
from typing import Literal

import numpy as np
from diffusers import StableDiffusionXLPipeline
from insightface.utils import face_align
from PIL import Image
import torch
import cv2


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_config_jsonable(config):
    """replaces all non-serializable objects in configuration to default value

    use with caution: hyper-parameters are supposed to be stored elsewhere
    """
    for k, v in config.items():
        if isinstance(v, dict):
            make_config_jsonable(v)
        try:
            json.dumps(v)
        except:
            config[k] = None
    return config


def get_faceid_embeds(
    pipe: StableDiffusionXLPipeline, 
    app, 
    img_path: str, 
    version: Literal[None, "plus, plus-v2"] = None,
    do_cfg=False,
    decoupled=False, 
    dcg_type=1,
    batch_size=1,
    adapt_det_size=False,
    dtype=torch.float16,
    drop_cond=False,
    return_bbox=False
): 
    """returns ip_adapter_image_embeds and sets clip_embeds inside pipe"""
    if isinstance(img_path, str):
        img_path = Image.open(img_path)
    image = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_BGR2RGB)
   
    if adapt_det_size:
        det_sizes = [(size, size) for size in range(640, 256, -64)]
        for size in det_sizes:
            app.det_model.input_size = size
            faces = app.get(image)
            if len(faces) > 0:
                break
    else:
        faces = app.get(image)

    if len(faces) == 0:
        return (None, None) if not return_bbox else (None, None, None)

    ref_images_embeds = []
    face_image = torch.from_numpy(faces[0].normed_embedding)
    ref_images_embeds.append(face_image.unsqueeze(0))
    ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)

    # get's splitted inside pipe.prepare_ip_adapter_image_embeds() code
    if do_cfg:
        neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
        id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=dtype, device=pipe.device)
    else:
        id_embeds = ref_images_embeds

    if version in ["plus", "plus-v2"]: # additionally need clip embeddings for plus and plus-v2 versions
        ip_adapter_images = []
        ip_adapter_images.append(face_align.norm_crop(image, landmark=faces[0].kps, image_size=224))
        
        if not decoupled:
            clip_embeds = pipe.prepare_ip_adapter_image_embeds(
                [ip_adapter_images], 
                None, 
                torch.device(pipe.device), 
                batch_size, 
                do_cfg)[0]
        else:
            # assert do_cfg # should be True
            clip_embeds = pipe.prepare_ip_adapter_image_embeds(
                [ip_adapter_images], 
                None, 
                torch.device(pipe.device), 
                batch_size, 
                do_cfg,
                do_decoupled_cfg=True,
                dcg_type=dcg_type)[0]
        # add clip_embeds to resampler
        pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=dtype)
        pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True if version == "plus-v2" else False
    else:
        clip_embeds = None

    if not return_bbox:
        return id_embeds, clip_embeds
    else:
        return id_embeds, clip_embeds, faces[0].bbox