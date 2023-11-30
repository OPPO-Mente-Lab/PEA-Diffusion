# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from typing import Callable, List, Optional, Union

from einops import rearrange
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)

from utils.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from utils.universal import UniversalCheckpoint
from utils.custom_dataset_sdxl import DataModuleCustom,BUCKETS

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

import open_clip
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel

from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip

NUM_blocks= 3
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

def getActivation(activation,name,residuals_present):
    # the hook signature
    if residuals_present:
        def hook(model, input, output):
            activation[name] = output[0]
    else:
        def hook(model, input, output):
            activation[name] = output
    return hook
    
def cast_hook(unet,dicts):
    for i in range(NUM_blocks):
        unet.down_blocks[i].register_forward_hook(getActivation(dicts,'d'+str(i),True))
    unet.mid_block.register_forward_hook(getActivation(dicts,'m',False))
    for i in range(NUM_blocks):
        unet.up_blocks[i].register_forward_hook(getActivation(dicts,'u'+str(i),False))


class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('SEA-Diffusion Module')
        parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
        return parent_parser
    def __init__(self, args):
        super().__init__()
        paths = args.text_encoder_path
        if args.text_encoder=="mul_clip":
            # paths = 'CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder.text.output_tokens = True
            self.proj = MLP(1024, 1280, 2048, 2048, use_residual=False)

        elif args.text_encoder=="chinese_clip":
            # paths = 'clip_cn_vit-h-14.pt'
            self.tokenizer = clip.tokenize
            self.text_encoder, preprocess = load_from_name(paths, download_root='../models')
            self.proj = MLP(1024, 1280, 1024, 2048, use_residual=False)

        elif args.text_encoder=="mt5":
            # paths = 'mt5-xl'
            self.text_encoder = T5EncoderModel.from_pretrained(paths)
            self.tokenizer = T5Tokenizer.from_pretrained(paths)
            self.proj = MLP(2048, 1280, 2048, 2048, use_residual=False)
            
        elif args.text_encoder=="alt_clip":
            # paths = '/data_share/data/multimodel_data/clip_model'
            loader = AutoLoader(
                task_name="txt_img_matching",
                model_name="AltCLIP-XLMR-L",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
                model_dir=paths
            )
            self.text_encoder = loader.get_model()
            self.tokenizer = loader.get_tokenizer()
            self.proj = MLP(768, 1280, 2048, 2048, use_residual=False)
        else: ## Mul clip + Chinese clip
            paths = 'CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder_mul, _, preprocess_mul = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer_mul = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder_mul.text.output_tokens = True

            paths = 'clip_cn_vit-h-14.pt'
            self.tokenizer_zh = clip.tokenize
            self.text_encoder_zh, preprocess_zh = load_from_name(paths, download_root='../models')
            self.proj = MLP(2048, 1280, 2048, 2048, use_residual=False)

        
        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
        self.test_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor)
        if args.load_ckpt_id:
            self.proj.load_state_dict(torch.load(os.path.join(args.load_ckpt_path, f"proj_0_{args.load_ckpt_id}/pytorch_model.bin"), map_location="cpu"))
        if args.KD:
            self.text_encoder_1 = CLIPTextModel.from_pretrained(f"{args.model_path}/text_encoder")
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(f"{args.model_path}/tokenizer")
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(f"{args.model_path}/text_encoder_2")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(f"{args.model_path}/tokenizer_2")
            self.unet_teacher = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")

            self.KD_teacher = {}
            self.KD_student= {}
            cast_hook(self.unet,self.KD_student)
            cast_hook(self.unet_teacher,self.KD_teacher)


    def setup(self, stage) -> None:
        if stage == 'fit':
            # 随便设置的，需要修改10^9/16/28=7,812,500
            self.total_steps = 2232142
            # self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = [{'params': self.proj.parameters()}]
        return configure_optimizers(self, model_params=model_params)
    
    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer_1, self.tokenizer_2] if self.tokenizer_1 is not None else [self.tokenizer_2]
        text_encoders = ([self.text_encoder_1, self.text_encoder_2] if self.text_encoder_1 is not None else [self.text_encoder_2])

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                # logger.warning(
                #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                #     f" {tokenizer.model_max_length} tokens: {removed_text}"
                # )

            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_tokens = [""]*batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds


    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.pipe.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.pipe.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            self.vae.to(dtype=torch.float32)
            latents = self.vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
            latents = latents.half() * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )

        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)
        with torch.no_grad():
            if args.text_encoder=="mul_clip":
                _,encoder_hidden_states = self.text_encoder.encode_text(batch["input_ids"])
                _,encoder_hidden_states_uncond = self.text_encoder.encode_text(batch["input_ids_uncond"])
            elif args.text_encoder=="chinese_clip":
                encoder_hidden_states,_ = self.text_encoder.encode_text(batch["input_ids"])
                encoder_hidden_states_uncond,_ = self.text_encoder.encode_text(batch["input_ids_uncond"])
            elif args.text_encoder=="mt5":

                # text_inputs = self.tokenizer(
                #         prompt,
                #         padding="max_length",
                #         max_length=77,
                #         truncation=True,
                #         return_tensors="pt",
                #     )
                # text_input_ids = text_inputs.input_ids
                pad_index = self.tokenizer.pad_token_id
                attention_mask = batch["input_ids"].ne(pad_index)
                text_embeddings = self.text_encoder.encoder(batch["input_ids"],attention_mask=attention_mask,output_hidden_states=True,)
                encoder_hidden_states = text_embeddings[0]

                attention_mask = batch["input_ids_uncond"].ne(pad_index)
                text_embeddings = self.text_encoder.encoder(batch["input_ids_uncond"],attention_mask=attention_mask,output_hidden_states=True,)
                encoder_hidden_states_uncond = text_embeddings[0]

            elif args.text_encoder=="alt_clip":
                tokenizer_out = self.tokenizer(
                    batch["instance_prompt_ids"],
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                text = tokenizer_out["input_ids"].to(latents.device)
                attention_mask = tokenizer_out["attention_mask"].to(latents.device)
                _,_ ,encoder_hidden_states= self.text_encoder.get_text_features(text, attention_mask=attention_mask)

                tokenizer_out = self.tokenizer(
                    [""]*bsz,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                text = tokenizer_out["input_ids"].to(latents.device)
                attention_mask = tokenizer_out["attention_mask"].to(latents.device)
                _,_,encoder_hidden_states_uncond = self.text_encoder.get_text_features(text, attention_mask=attention_mask)
            else:
                input_ids = self.tokenizer_mul(batch["instance_prompt_ids"],context_length=64).to(latents.device)
                input_ids_uncond = self.tokenizer_mul([""]*bsz,context_length=64).to(latents.device)
                _,encoder_hidden_states_mul = self.text_encoder_mul.encode_text(input_ids)
                _,encoder_hidden_states_uncond_mul = self.text_encoder_mul.encode_text(input_ids_uncond)

                encoder_hidden_states_zh = self.text_encoder_zh.encode_text(batch["input_ids"])
                encoder_hidden_states_uncond_zh = self.text_encoder_zh.encode_text(batch["input_ids_uncond"])
                encoder_hidden_states = torch.cat([encoder_hidden_states_mul,encoder_hidden_states_zh],-1)
                encoder_hidden_states_uncond = torch.cat([encoder_hidden_states_uncond_mul,encoder_hidden_states_uncond_zh],-1)

        add_text_embeds,encoder_hidden_states = self.proj(encoder_hidden_states) ## B*77*1024 --> B*1280   B*77*2048  
        add_text_embeds_uncond,encoder_hidden_states_uncond = self.proj(encoder_hidden_states_uncond)

        crops_coords_top_left = batch["crops_coords_top_left"]
        original_size = batch["original_size"]
        target_size = torch.tensor([BUCKETS[batch["bucket_id"]]]*len(batch["crops_coords_top_left"]),device=latents.device)
        add_time_ids = torch.cat([original_size,crops_coords_top_left,target_size],1) ##
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        uncond = 0.1
        random = torch.rand(latents.size(0), device=latents.device)
        prompt_mask = rearrange(random < uncond, "n -> n 1 1")
        encoder_hidden_states = torch.where(prompt_mask, encoder_hidden_states_uncond, encoder_hidden_states)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states,added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        if args.KD and args.hybrid_training:
            ## Chinese or English tags in batch
            zh_or_not = batch["zh_or_not"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            loss = loss*zh_or_not
        loss = loss.mean([1, 2, 3]).mean()
        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

        if args.KD:
            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds,pooled_prompt_embeds = self.encode_prompt(batch["texts_en"],latents.device)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                prompt_embeds = torch.where(prompt_mask, negative_prompt_embeds, prompt_embeds)
                ## noisy_latents = torch.nn.UpsamplingBilinear2d(scale_factor=2)(noisy_latents)
                noise_pred_teacher = self.unet_teacher(noisy_latents, timesteps, prompt_embeds,added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
            if args.hybrid_training:
                loss_teacher = (F.mse_loss(noise_pred, noise_pred_teacher, reduction="none")*(1-zh_or_not)).mean([1, 2, 3]).mean()
            else:
                loss_teacher = F.mse_loss(noise_pred, noise_pred_teacher, reduction="none").mean([1, 2, 3]).mean()
            self.log("train_loss_logits", loss_teacher.item(),  on_epoch=False, prog_bar=True, logger=True)
            loss += loss_teacher

            loss_features=0
            ## latent.shape = B*4*88*176
            for i in range(NUM_blocks): # B*320*44*88  B*640*22*44 B*1280*22*44
                down_feature = F.mse_loss(self.KD_teacher['d'+str(i)],self.KD_student['d'+str(i)], reduction="none")
                if args.hybrid_training:
                    down_feature = down_feature*(1-zh_or_not)
                loss_features=loss_features+down_feature.mean([1, 2, 3]).mean()
            middle_feature = F.mse_loss(self.KD_teacher['m'],self.KD_student['m'], reduction="none")  # B*1280*22*44
            if args.hybrid_training:
                middle_feature = middle_feature*(1-zh_or_not) 
            loss_features=loss_features+middle_feature.mean([1, 2, 3]).mean()
            for i in range(NUM_blocks): # B*1280*44*88 B*640*88*176 B 2*320*88*176
                up_feature = F.mse_loss(self.KD_teacher['u'+str(i)],self.KD_student['u'+str(i)], reduction="none")
                if args.hybrid_training:
                    up_feature = up_feature*(1-zh_or_not)
                loss_features=loss_features+up_feature.mean([1, 2, 3]).mean()
            self.log("train_loss_features", loss_features.item(),  on_epoch=False, prog_bar=True, logger=True)

            loss += loss_features*0.1

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % args.every_n_steps == 0:
                print('saving model...')
                save_directory = os.path.join(args.default_root_dir,f'proj_{self.global_step}')
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.proj.state_dict(), os.path.join(save_directory,"pytorch_model.bin"))
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        pass
    
    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = DataModuleCustom.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    tokenizer = model.tokenizer
    datamoule = DataModuleCustom(args, tokenizer=tokenizer)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,callbacks=[lr_monitor,checkpoint_callback])

    trainer.fit(model, datamoule)
