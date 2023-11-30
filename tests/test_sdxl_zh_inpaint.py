# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image

from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel,DPMSolverMultistepScheduler,EulerDiscreteScheduler
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.image_processor import VaeImageProcessor,PipelineImageInput
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import open_clip
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
# from flagai.auto_model.auto_loader import AutoLoader


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

def retrieve_latents(encoder_output, generator):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



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

    def __init__(self, model_id,proj_path,device):
        super().__init__()
        if TEXT_ENCODER=="chinese":
            self.text_encoder, preprocess = load_from_name("/data_share/data/multimodel_data/clip_model/clip_cn_vit-h-14.pt", download_root='../models')
            self.proj = MLP(1024, 1280, 2048, 2048, use_residual=False).to(device).half() 
            self.tokenizer = clip.tokenize
            self.text_encoder = self.text_encoder.to(device)

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.pipe.vae_scale_factor)
        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.pipe.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.denoising_start = None
        # self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("/public_data/ma/models/stable-diffusion-xl-base-1.0/text_encoder_2")

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if TEXT_ENCODER == "chinese":
            text_input_ids = self.tokenizer(prompt).to(device)
            text_embeddings = self.text_encoder.encode_text(text_input_ids)
        elif TEXT_ENCODER=="clip":
            text_input_ids = self.tokenizer(prompt).to(device)
            _,text_embeddings = self.text_encoder.encode_text(text_input_ids)

        elif TEXT_ENCODER=="mt5":

            text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids.to(device)
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
            text_input_ids = tokenizer_out["input_ids"].to(device)
            attention_mask = tokenizer_out["attention_mask"].to(device)
            _,_ ,text_embeddings= self.text_encoder.get_text_features(text_input_ids, attention_mask=attention_mask)

        elif TEXT_ENCODER=="mul_clip—chinese_clip":
            text_input_ids = self.tokenizer_mul(prompt,context_length=64).to(device)
            _,text_embeddings_mul = self.text_encoder_mul.encode_text(text_input_ids)

            text_input_ids = self.tokenizer_zh(prompt).to(device)
            text_embeddings_zh = self.text_encoder_zh.encode_text(text_input_ids)

            text_embeddings = torch.cat([text_embeddings_mul,text_embeddings_zh],-1)

        else:
            text_input_ids = self.tokenizer(prompt, context_length=32).to(device)
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
                uncond_input_ids = self.tokenizer(uncond_tokens).to(device)
                uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            elif TEXT_ENCODER=="clip":
                uncond_input_ids = self.tokenizer(uncond_tokens).to(device)
                _,uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            elif TEXT_ENCODER=="mt5":

                text_inputs = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids.to(device)
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
                text_input_ids = tokenizer_out["input_ids"].to(device)
                attention_mask = tokenizer_out["attention_mask"].to(device)
                _,_,uncond_embeddings = self.text_encoder.get_text_features(text_input_ids, attention_mask=attention_mask)
            elif TEXT_ENCODER=="mul_clip—chinese_clip":
                text_input_ids = self.tokenizer_mul(uncond_tokens,context_length=64).to(device)
                _,uncond_embeddings_mul = self.text_encoder_mul.encode_text(text_input_ids)

                text_input_ids = self.tokenizer_zh(uncond_tokens).to(device)
                uncond_embeddings_zh = self.text_encoder_zh.encode_text(text_input_ids)

                uncond_embeddings = torch.cat([uncond_embeddings_mul,uncond_embeddings_zh],-1)
            else:
                uncond_input_ids = self.tokenizer(uncond_tokens, context_length=32).to(device)
                _,uncond_embeddings = self.text_encoder.encode_text(uncond_input_ids)
            add_text_embeds_uncond,uncond_embeddings_2048 = self.proj(uncond_embeddings)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings_2048.shape[1]
            uncond_embeddings_2048 = uncond_embeddings_2048.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings_2048 = uncond_embeddings_2048.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings_cat = torch.cat([uncond_embeddings_2048, text_embeddings_2048])
            add_text_embeds = torch.cat([add_text_embeds_uncond, add_text_embeds])

        return text_embeddings_cat,add_text_embeds


    # def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
    #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
    #     add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    #     return add_time_ids

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.pipe.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.pipe.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.pipe.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.pipe.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.half().encode(image), generator=generator)

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        add_noise=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
        image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = 1024, #64*40
        width: Optional[int] = 1024, #64*40
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
  
        # 1. Check inputs
        do_classifier_free_guidance = guidance_scale > 1.0


        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)

        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.pipe.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if self.denoising_start is None else False
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.pipe.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.pipe.unet.config} expects"
                    f" {self.pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.pipe.unet.__class__} should have either 4 or 9 input channels, not {self.pipe.unet.config.in_channels}."
            )
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:]
        height = height * self.pipe.vae_scale_factor
        width = width * self.pipe.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        # if self.text_encoder_2 is None:
        #     text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        # else:
        #     text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        text_encoder_projection_dim = 1280
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        denoising_end = None
        if (
            denoising_end is not None
            and self.denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(self.denoising_start)
            and self.denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 11.1 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # 8. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="np")

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        return image


if __name__ == '__main__':
    TEXT_ENCODER = "chinese" ## clip  chinese mt5 alt_clip mul_clip—chinese_clip
    device = "cuda"
    RESUME_PATH = "/public_data/ma/code/text2img/result/stablediffusion_distill_zh"
    RESUME_ID = 49999  

    proj_path = os.path.join(RESUME_PATH, f"proj_0_{RESUME_ID}/pytorch_model.bin")
    model_id = "/public_data/wrc/models/stable-diffusion-xl-1.0-inpainting-0.1"
    sdt = StableDiffusionTest(model_id,proj_path,device)
    negative_prompt="低分辨率、低质量、混乱，黑暗环境"

    batch=4
    while True:
        raw_text = input("\nPlease Input Query (stop to exit) >>> ")
        if not raw_text:
            print('Query should not be empty!')
            continue
        if raw_text == "stop":
            break

        img_url = "/public_data/ma/code/text2img/outputs/inpaint0.png"
        mask_url = "/public_data/ma/code/text2img/outputs/inpaint1.png"

        image = load_image(img_url).resize((1024, 1024))
        mask_image = load_image(mask_url).resize((1024, 1024))

        images = sdt([raw_text]*batch,image = image,mask_image =mask_image, negative_prompt=[negative_prompt]*batch)
        for i, image in enumerate(images):
            image.save(f"outputs/{i}_new.png")
        grid = image_grid(images, rows=1, cols=batch)
        grid.save("outputs/1.png")
