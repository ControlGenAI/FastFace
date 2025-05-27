from typing import Any, Dict, List, Optional, Tuple, Union
import os

from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler
)
from diffusers.models import ImageProjection
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .utils import get_faceid_embeds
from .dcg import decoupled_cfg_predict


class DecoupledGuidancePipelineXL(StableDiffusionXLPipeline):
    """pipeline to be used together with IpAdapter, adds separate image and textual guidance"""
    def prepare_ip_adapter_image_embeds(
        self, 
        ip_adapter_image, 
        ip_adapter_image_embeds, 
        device, 
        num_images_per_prompt, 
        do_classifier_free_guidance,
        do_decoupled_cfg,
        dcg_type=1,
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                
                if not do_decoupled_cfg:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
                else: # add additional img embedding
                    embed = single_negative_image_embeds if dcg_type in [1, 2] else single_image_embeds
                    single_image_embeds = torch.cat([
                        single_negative_image_embeds,
                        embed, 
                        single_image_embeds], 
                    dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds
    
    def preprocess_prompt_embeds(
        self,
        dcg_type,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        add_text_embeds,
        negative_add_time_ids,
        add_time_ids
    ):
        if dcg_type == 1:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                prompt_embeds,
                prompt_embeds],
            dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids, add_time_ids], dim=0)
        elif dcg_type == 2:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                prompt_embeds,
                negative_prompt_embeds],
            dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, negative_pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids, negative_add_time_ids], dim=0)
        elif dcg_type == 3:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                negative_prompt_embeds,
                prompt_embeds,
            ], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, negative_add_time_ids, add_time_ids], dim=0)

        return prompt_embeds, add_text_embeds, add_time_ids

    @torch.no_grad()
    def execute(
        self,
        face_image,
        prompt,
        generator,
        pipe_kwargs,
        after_hook_fn=None
    ):
        id_embeds, _ = get_faceid_embeds(
            self,
            self.app,
            face_image, 
            "plus-v2", 
            do_cfg=True,
            decoupled=True,
            dcg_type=pipe_kwargs["dcg_kwargs"]["dcg_type"]
        )

        res = self(
            prompt=prompt,
            ip_adapter_image_embeds=[id_embeds],
            num_images_per_prompt=1,
            generator=generator,
            **pipe_kwargs
        ).images[0]
        
        if after_hook_fn is not None:
            after_hook_fn(self, res)
        
        return res

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,

        guidance_scale_a: float = 5.0, 
        guidance_scale_b: float = 5.0,
        dcg_kwargs: Dict = {"dcg_type": 3},

        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = 999. # arbitrary number to always enable do_cfg inside pipeline
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        dcg_type=dcg_kwargs["dcg_type"]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )        

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds, add_text_embeds, add_time_ids = self.preprocess_prompt_embeds(
                dcg_type,
                negative_prompt_embeds,
                prompt_embeds,
                negative_pooled_prompt_embeds,
                add_text_embeds,
                negative_add_time_ids,
                add_time_ids
            )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # Encode face image for ip_adapter
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
                do_decoupled_cfg=True,
                dcg_type=dcg_type
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        if hasattr(self, "track_norms") and self.track_norms:
            self.norms = [[], [], [], [], []]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                # NOTE: expanding to 3 to enable decoupled guidance
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:                 
                    dcg_out = decoupled_cfg_predict(
                        noise_pred,
                        guidance_scale_a,
                        guidance_scale_b,
                        (i, t),
                        **dcg_kwargs
                    )
                    noise_pred = dcg_out["pred"]
                    if hasattr(self, "track_norms") and self.track_norms:
                        for idx, n in enumerate(self.norms):
                            n.append(dcg_out["norms"][idx])

                # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def prepare_dcg_pipeline(config, device, *args, **kwargs):
    model_name = config["model_name"]
    pipe_kwargs = config["pipe_kwargs"]
    n_steps = pipe_kwargs["num_inference_steps"]
    adaptation_method = config["method"]

    method2pipeline = {
        "faceid": DecoupledGuidancePipelineXL, # currently support only faceid-plusv2
    }

    if adaptation_method not in method2pipeline:
        raise ValueError(f"Face adaptation method {adaptation_method} not recongized, supported values: {str(list(method2pipeline.keys()))}")

    if model_name == "base":
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs
        ).to(device)
    if model_name == "hyper":
        if n_steps not in [1, 2, 4, 8]:
            raise ValueError(f"num_inference_steps={n_steps} not supported for hyper checkpoint")
        
        if not os.path.exists(f"models_cache/hyper-{n_steps}-pipe-fused"):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                variant="fp16", 
                torch_dtype=torch.float16, 
                cache_dir="models_cache/sdxl"
            ).to(device)
            pipe.load_lora_weights(hf_hub_download(
                "ByteDance/Hyper-SD", 
                f"Hyper-SDXL-{n_steps}step{'s' if n_steps > 1 else ''}-lora.safetensors", 
                cache_dir=f"models_cache/sdxl-hyper-{n_steps}")
            )
            pipe.fuse_lora()
            pipe.unload_lora_weights()

            # save pipeline for later use
            pipe.save_pretrained(f"models_cache/hyper-{n_steps}-pipe-fused")

        pipe = method2pipeline[adaptation_method].from_pretrained(
            f"models_cache/hyper-{n_steps}-pipe-fused",
            torch_dtype=torch.float16,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs,
        ).to(device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif model_name == "lcm":
        if not os.path.exists("models_cache/lcm-lora-fused"):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                variant="fp16", 
                torch_dtype=torch.float16, 
                cache_dir="models_cache/sdxl"
            ).to(device)

            # let's merge lora weights and save this model
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", cache_dir="models_cache/lcm-lora")
            pipe.fuse_lora()
            pipe.unload_lora_weights()

            # save pipeline for later use
            pipe.save_pretrained("models_cache/lcm-lora-fused")
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "models_cache/lcm-lora-fused",  
            torch_dtype=torch.float16,
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif model_name == "lightning":
        if n_steps not in [2, 4, 8]:
            raise ValueError(f"n_steps={n_steps} not supported for ligntning checkpoint")
        unet = UNet2DConditionModel.from_config(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="unet",
            cache_dir="models_cache/sdxl").to(device, torch.float16)
        
        unet.load_state_dict(load_file(
            hf_hub_download(
                "ByteDance/SDXL-Lightning", 
                f"sdxl_lightning_{n_steps}step_unet.safetensors",
                cache_dir=f"models_cache/sdxl-lightning-{n_steps}"), 
                device="cpu"
            )
        )
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            unet=unet, 
            torch_dtype=torch.float16,
            cache_dir="models_cache/sdxl",
            variant="fp16",
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing"
        )
    elif model_name == "turbo":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/sdxl-turbo", subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            subfolder="unet",
            cache_dir="models_cache/sdxl-turbo"
        ).to(device)    
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            unet=unet,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = scheduler
    
    return pipe


def get_faceid_pipeline(config, device):
    os.makedirs("models_cache", exist_ok=True)
    
    pipe = prepare_dcg_pipeline(config, device)
    ip_adapter_scale = config.get("ip_adapter_scale", 0.5)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=torch.float16,
        cache_dir="models_cache/faceid_img_encoder",
    ).to(device)

    if ("t2i" not in config) or (not config["t2i"]):
        pipe.register_modules(image_encoder=image_encoder)

        clip_image_size = pipe.image_encoder.config.image_size
        feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
        pipe.register_modules(feature_extractor=feature_extractor)

        pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID", 
            subfolder=None, 
            weight_name="ip-adapter-faceid-plusv2_sdxl.bin", 
            image_encoder_folder=None,
            cache_dir="models_cache/ipadapter"
        )
        
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        pipe.track_norms=False

    device_id = device.index if hasattr(device, 'index') else 0
    print("DEVICE ID: ", device_id)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], provider_options=[{"device_id": device_id}, {}])
    app.prepare(ctx_id=0, det_size=(640, 640))
    pipe.app = app

    return pipe


name2pipe = {
    "faceid": get_faceid_pipeline,
}