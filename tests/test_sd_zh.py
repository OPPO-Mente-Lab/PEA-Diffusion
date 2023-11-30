# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image

from diffusers import AutoencoderKL, StableDiffusionPipeline,DPMSolverMultistepScheduler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import open_clip
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from diffusers.image_processor import VaeImageProcessor

class MLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, hidden_dim=2048):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
    ## B*77*1024 -->  B*77*768  
    def forward(self, x):
        x = self.layernorm(x)
        x = self.projector(x)
        return x
        
class StableDiffusionTest():

    def __init__(self, model_id,proj_path,DEVICE):
        super().__init__()

        if TEXT_ENCODER=="mul_clip":
            paths = '/data_share/data/multimodel_data/clip_model/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder.text.output_tokens = True
            self.proj = MLP(1024,768, 3072).to(DEVICE)

        elif TEXT_ENCODER=="chinese_clip":
            paths = '/data_share/data/multimodel_data/clip_model/clip_cn_vit-h-14.pt'
            self.tokenizer = clip.tokenize
            self.text_encoder, preprocess = load_from_name(paths, download_root='../models')
            self.proj = MLP(1024,768, 3072).to(DEVICE).half()
            # self.proj = MLP(1024,1024, 2048)

        self.text_encoder = self.text_encoder.to(DEVICE)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(DEVICE)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler,torch_dtype=torch.float16).to(DEVICE) 
        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)


    def encode_prompt(self, prompt, DEVICE, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if TEXT_ENCODER=="mul_clip":
            text_input_ids = self.tokenizer(prompt).to(DEVICE)
            _,text_embeddings = self.text_encoder.encode_text(text_input_ids)
        elif TEXT_ENCODER=="chinese_clip":
            text_input_ids = self.tokenizer(prompt).to(DEVICE)
            text_embeddings,_ = self.text_encoder.encode_text(text_input_ids)
            
        text_embeddings_1024 = self.proj(text_embeddings)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            # uncond_input_ids = self.tokenizer(uncond_tokens).to(DEVICE)
            # uncond_embeddings,_ = self.text_encoder.encode_text(uncond_input_ids)

            if TEXT_ENCODER=="mul_clip":
                input_ids_uncond = self.tokenizer(uncond_tokens).to(DEVICE)
                _,uncond_embeddings = self.text_encoder.encode_text(input_ids_uncond)
            elif TEXT_ENCODER=="chinese_clip":
                input_ids_uncond = self.tokenizer(uncond_tokens).to(DEVICE)
                uncond_embeddings,_ = self.text_encoder.encode_text(input_ids_uncond)
                
            uncond_embeddings_2048 = self.proj(uncond_embeddings)

            text_embeddings_cat = torch.cat([uncond_embeddings_2048, text_embeddings_1024])

        return text_embeddings_cat


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 1024, #64*40
        width: Optional[int] = 1024, #64*22
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        DEVICE = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
 
        prompt_embeds = self.encode_prompt(prompt, DEVICE, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        prompt_embeds = prompt_embeds.half()

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            DEVICE,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)


        # 7. Denoising loop
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)


            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            #latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # 8. Post-processing
        # image = self.vae.decode(latents.float() / self.vae.config.scaling_factor, return_dict=False)[0]
        # image = self.image_processor.postprocess(image, output_type="np")
        image = self.pipe.decode_latents(latents)
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image


if __name__ == '__main__':
    DEVICE = "cuda"
    TEXT_ENCODER = "chinese_clip"  ## mul_clip  chinese_clip  mt5  alt_clip
    OUTPUTS = "./output"
    RESUME_PATH = "/public_data/ma/code/multilingual_stablediffusion/result/stablediffusion_distill_zh_sd1" 
    RESUME_ID = "41999"

    proj_path = os.path.join(RESUME_PATH, f"proj_0_{RESUME_ID}/pytorch_model.bin")
    model_id = "/public_data/ma/models/Realistic_Vision_V1.4"
    sdt = StableDiffusionTest(model_id,proj_path,DEVICE)

    negative_prompt=""
    batch=4
    while True:
        raw_text = input("\nPlease Input Query (stop to exit) >>> ")
        if not raw_text:
            print('Query should not be empty!')
            continue
        if raw_text == "stop":
            break
        images = sdt([raw_text]*batch,negative_prompt=[negative_prompt]*batch)
        for i, image in enumerate(images):
            image.save(f"{OUTPUTS}/{i}_new.png")
        grid = image_grid(images, rows=1, cols=batch)
        grid.save("{OUTPUTS}/1.png")


