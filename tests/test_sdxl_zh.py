# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import sys
import random
import json
from tqdm import tqdm

from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLPipeline,DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

import torch
from torch.nn import functional as F
import torch.nn as nn

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import open_clip
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class MLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1280, hidden_dim=2048, out_dim1=2048, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
    ## B*77*1024 --> B*1280   B*77*2048  
    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x2 = self.fc(x2)
        if self.use_residual:
            x = x + residual
        x1 = torch.mean(x,1)
        return x1,x2

class StableDiffusionTest():

    def __init__(self, model_id,proj_path,DEVICE,DOWNSTREAM):
        super().__init__()
        if TEXT_ENCODER=="chinese":
            self.text_encoder, preprocess = load_from_name("/data_share/data/multimodel_data/clip_model/clip_cn_vit-h-14.pt", download_root='../models')
            self.proj = MLP(1024, 1280, 2048, 2048, use_residual=False).to(DEVICE).half()  
            self.tokenizer = clip.tokenize
            self.text_encoder = self.text_encoder.to(DEVICE)

        elif TEXT_ENCODER=="clip":
            paths = '/data_share/data/multimodel_data/clip_model/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder.text.output_tokens = True
            self.proj = MLP(1024, 1280, 1024, 2048, use_residual=False).to(DEVICE)

        elif TEXT_ENCODER=="mt5":
            self.text_encoder = T5EncoderModel.from_pretrained("/data_share/data/multimodel_data/clip_model/mt5-xl")
            self.tokenizer = T5Tokenizer.from_pretrained("/data_share/data/multimodel_data/clip_model/mt5-xl")
            self.proj = MLP(2048, 1280, 2048, 2048, use_residual=False).to(DEVICE)

        elif TEXT_ENCODER=="alt_clip":
            loader = AutoLoader(
                task_name="txt_img_matching",
                model_name="AltCLIP-XLMR-L",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
                model_dir="/data_share/data/multimodel_data/clip_model"
            )
            self.text_encoder = loader.get_model().to(DEVICE)
            self.tokenizer = loader.get_tokenizer()
            self.proj = MLP(768, 1280, 2048, 2048, use_residual=False).to(DEVICE)

        elif TEXT_ENCODER=="mul_clip—chinese_clip":

            paths = '/data_share/data/multimodel_data/clip_model/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder_mul, _, preprocess_mul = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer_mul = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder_mul.text.output_tokens = True

            paths = '/data_share/data/multimodel_data/clip_model/clip_cn_vit-h-14.pt'
            self.tokenizer_zh = clip.tokenize
            self.text_encoder_zh, preprocess_zh = load_from_name(paths, download_root='../models')

            self.proj = MLP(2048, 1280, 2048, 2048, use_residual=False).to(DEVICE)
            self.text_encoder_zh = self.text_encoder_zh.to(DEVICE)
            self.text_encoder_mul = self.text_encoder_mul.to(DEVICE)

        else:
            resume='/public_data/ma/models/clip_zh/epoch_rank0_3_20230724.pt'
            model_name='wukong-large'
            self.text_encoder=load_model(resume,model_name).to(DEVICE)
            self.text_encoder.eval()

            self.tokenizer = tokenize
            self.proj = MLP(768, 1280, 1024, 2048, use_residual=False).to(DEVICE)

            self.text_encoder = self.text_encoder.to(DEVICE)

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(DEVICE)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler,torch_dtype=torch.float16).to(DEVICE) 

        if DOWNSTREAM=="LoRA":
            self.pipe.load_lora_weights("/public_data/ma/stable_models/csal_scenery_XL.safetensors")


        self.image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)
        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu"))

    def encode_prompt(self, prompt, DEVICE, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if TEXT_ENCODER == "chinese":
            text_input_ids = self.tokenizer(prompt).to(DEVICE)
            # text_embeddings = self.text_encoder.encode_text(text_input_ids)
            text_embeddings,_ = self.text_encoder.encode_text(text_input_ids)
        elif TEXT_ENCODER=="clip":
            text_input_ids = self.tokenizer(prompt).to(DEVICE)
            _,text_embeddings = self.text_encoder.encode_text(text_input_ids)

        elif TEXT_ENCODER=="mt5":

            text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids.to(DEVICE)
            pad_index = self.tokenizer.pad_token_id
            attention_mask = text_input_ids.ne(pad_index)
            text_embeddings = self.text_encoder.encoder(text_input_ids,attention_mask=attention_mask,output_hidden_states=True,)
            text_embeddings = text_embeddings[0]

        elif TEXT_ENCODER=="alt_clip":
            tokenizer_out = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = tokenizer_out["input_ids"].to(DEVICE)
            attention_mask = tokenizer_out["attention_mask"].to(DEVICE)
            _,_ ,text_embeddings= self.text_encoder.get_text_features(text_input_ids, attention_mask=attention_mask)

        elif TEXT_ENCODER=="mul_clip—chinese_clip":
            text_input_ids = self.tokenizer_mul(prompt,context_length=64).to(DEVICE)
            _,text_embeddings_mul = self.text_encoder_mul.encode_text(text_input_ids)

            text_input_ids = self.tokenizer_zh(prompt).to(DEVICE)
            text_embeddings_zh = self.text_encoder_zh.encode_text(text_input_ids)

            text_embeddings = torch.cat([text_embeddings_mul,text_embeddings_zh],-1)

        else:
            text_input_ids = self.tokenizer(prompt, context_length=32).to(DEVICE)
            _,text_embeddings = self.text_encoder.encode_text(text_input_ids)


            
        add_text_embeds,text_embeddings_2048 = self.proj(text_embeddings)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

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
            if TEXT_ENCODER == "chinese":
                uncond_input_ids = self.tokenizer(uncond_tokens).to(DEVICE)
                # uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
                uncond_embeddings,_ = self.text_encoder.encode_text(uncond_input_ids)
            elif TEXT_ENCODER=="clip":
                uncond_input_ids = self.tokenizer(uncond_tokens).to(DEVICE)
                _,uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            elif TEXT_ENCODER=="mt5":

                text_inputs = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids.to(DEVICE)
                pad_index = self.tokenizer.pad_token_id
                attention_mask = text_input_ids.ne(pad_index)
                text_embeddings = self.text_encoder.encoder(text_input_ids,attention_mask=attention_mask,output_hidden_states=True,)
                uncond_embeddings = text_embeddings[0]
            elif TEXT_ENCODER=="alt_clip":

                tokenizer_out = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = tokenizer_out["input_ids"].to(DEVICE)
                attention_mask = tokenizer_out["attention_mask"].to(DEVICE)
                _,_,uncond_embeddings = self.text_encoder.get_text_features(text_input_ids, attention_mask=attention_mask)
            elif TEXT_ENCODER=="mul_clip—chinese_clip":
                text_input_ids = self.tokenizer_mul(uncond_tokens,context_length=64).to(DEVICE)
                _,uncond_embeddings_mul = self.text_encoder_mul.encode_text(text_input_ids)

                text_input_ids = self.tokenizer_zh(uncond_tokens).to(DEVICE)
                uncond_embeddings_zh = self.text_encoder_zh.encode_text(text_input_ids)

                uncond_embeddings = torch.cat([uncond_embeddings_mul,uncond_embeddings_zh],-1)
            else:
                uncond_input_ids = self.tokenizer(uncond_tokens, context_length=32).to(DEVICE)
                _,uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            add_text_embeds_uncond,uncond_embeddings_2048 = self.proj(uncond_embeddings)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings_2048.shape[1]
            uncond_embeddings_2048 = uncond_embeddings_2048.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings_2048 = uncond_embeddings_2048.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings_cat = torch.cat([uncond_embeddings_2048, text_embeddings_2048])
            add_text_embeds = torch.cat([add_text_embeds_uncond, add_text_embeds])

        return text_embeddings_cat,add_text_embeds


    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


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
        # self.pipe.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        DEVICE = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
 
        prompt_embeds,add_text_embeds = self.encode_prompt(prompt, DEVICE, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        prompt_embeds = prompt_embeds.half()
        add_text_embeds = add_text_embeds.half()

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

        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        add_time_ids = add_time_ids.to(DEVICE).repeat(batch_size * num_images_per_prompt, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # 7. Denoising loop
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ]
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if not use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()
        
        # 8. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="np")

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image


if __name__ == '__main__':
    DEVICE = "cuda"
    TEXT_ENCODER = "chinese" ## clip  chinese mt5 alt_clip mul_clip—chinese_clip
    DOWNSTREAM = "" ## SSD LoRA
    RESUME_PATH = "/public_data/ma/code/text2img/result/stablediffusion_distill_zh"
    RESUME_ID = 49999  
    OUTPUTS = "./output"


    if DOWNSTREAM=="LoRA":
        model_id = "/public_data/ma/models/stable-diffusion-xl-base-1.0"
    elif DOWNSTREAM=="SSD":
        model_id = "/public_data/ma/models/SSD-1B"
    else:
        model_id = "/public_data/ma/models/stable-diffusion-xl-protovisionXL"

    proj_path = os.path.join(RESUME_PATH, f"proj_0_{RESUME_ID}/pytorch_model.bin")
    sdt = StableDiffusionTest(model_id,proj_path,DEVICE,DOWNSTREAM)

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
        grid.save(f"{OUTPUTS}/1.png")
