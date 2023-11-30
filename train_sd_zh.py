# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import torch.nn as nn
from einops import rearrange
import inspect
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
from utils.custom_dataset import DataModuleCustom

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler,DPMSolverMultistepScheduler
from torch.nn import functional as F

from typing import Callable, List, Optional, Union
from torchvision.utils import save_image
import open_clip
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip

NUM_blocks= 4
class MLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, hidden_dim=2048):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
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
        parser = parent_parser.add_argument_group('OPPO Stable Diffusion Module')
        parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
        return parent_parser
    def __init__(self, args):
        super().__init__()

        if args.text_encoder=="mul_clip":
            paths = 'CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/open_clip_pytorch_model.bin'
            self.text_encoder, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained=paths)
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
            self.text_encoder.text.output_tokens = True
            self.proj = MLP(1024, 1280, 2048, 2048, use_residual=False)

        elif args.text_encoder=="chinese_clip":
            paths = 'clip_cn_vit-h-14.pt'
            self.tokenizer = clip.tokenize
            self.text_encoder, preprocess = load_from_name(paths, download_root='../models')
            self.proj = MLP(1024,768, 2048)

        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
        self.test_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.save_hyperparameters(args)

        if args.load_ckpt_id:
            self.proj.load_state_dict(torch.load(
                os.path.join(args.load_ckpt_path, f"proj_0_{args.load_ckpt_id}/pytorch_model.bin"), map_location="cpu"))
        if args.KD:
            self.text_encoder_1 = CLIPTextModel.from_pretrained(f"{args.model_path}/text_encoder")
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(f"{args.model_path}/tokenizer")
            self.unet_teacher = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
            self.KD_teacher = {}
            self.KD_student= {}
            
            cast_hook(self.unet,self.KD_student)
            cast_hook(self.unet_teacher,self.KD_teacher)


    def setup(self, stage) -> None:
        if stage == 'fit':
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

        # textual inversion: procecss multi-vector tokens if necessary
        text_inputs = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_1(text_input_ids.to(device))
        prompt_embeds = prompt_embeds[0]

        # bs_embed, seq_len, _ = prompt_embeds.shape
        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        uncond_tokens = [""]*batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer_1(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids
        uncond_embeddings = self.text_encoder_1(uncond_input_ids.to(device))
        uncond_embeddings = uncond_embeddings[0]

        return prompt_embeds, uncond_embeddings

    def training_step(self, batch, batch_idx):

        # self.unet.train()
        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents* self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
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

        encoder_hidden_states = self.proj(encoder_hidden_states) ## B*77*1024 --> B*77*768  
        encoder_hidden_states_uncond = self.proj(encoder_hidden_states_uncond)

        uncond = 0.1
        random = torch.rand(latents.size(0), device=latents.device)
        prompt_mask = rearrange(random < uncond, "n -> n 1 1")
        encoder_hidden_states = torch.where(prompt_mask, encoder_hidden_states_uncond, encoder_hidden_states)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

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
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(batch["texts_en"],latents.device)
                prompt_embeds = torch.where(prompt_mask, negative_prompt_embeds, prompt_embeds)
                noise_pred_teacher = self.unet_teacher(noisy_latents, timesteps, prompt_embeds, return_dict=False)[0]
            if args.hybrid_training:
                loss_teacher = (F.mse_loss(noise_pred, noise_pred_teacher, reduction="none")*(1-zh_or_not)).mean([1, 2, 3]).mean()
            else:
                loss_teacher = F.mse_loss(noise_pred, noise_pred_teacher, reduction="none").mean([1, 2, 3]).mean()
            self.log("train_loss_logits", loss_teacher.item(),  on_epoch=False, prog_bar=True, logger=True)
            loss += loss_teacher

            loss_features=0
            ## latent.shape = B*4*88*176
            for i in range(NUM_blocks): # B*320*4*4  B*640*2*2 B*1280*1*1  B*1280*1*1
                down_feature = F.mse_loss(self.KD_teacher['d'+str(i)],self.KD_student['d'+str(i)], reduction="none")
                if args.hybrid_training:
                    down_feature = down_feature*(1-zh_or_not)

                if not (torch.isinf(down_feature).any() or torch.isnan(down_feature).any()):
                    loss_features=loss_features+down_feature.mean([1, 2, 3]).mean()
                else:
                    print(f"down_feature:{i}")

            middle_feature = F.mse_loss(self.KD_teacher['m'],self.KD_student['m'], reduction="none")  # B*1280*22*44
            if args.hybrid_training:
                middle_feature = middle_feature*(1-zh_or_not) 

            if not (torch.isinf(middle_feature).any() or torch.isnan(middle_feature).any()):
                loss_features=loss_features+middle_feature.mean([1, 2, 3]).mean()
            else:
                print("middle_feature")

            for i in range(NUM_blocks): # B*1280*2*2  B*1280*4*4  B*640*8*8   B*320*8*8
                up_feature = F.mse_loss(self.KD_teacher['u'+str(i)],self.KD_student['u'+str(i)], reduction="none")
                if args.hybrid_training:
                    up_feature = up_feature*(1-zh_or_not)

                if not (torch.isinf(up_feature).any() or torch.isnan(up_feature).any()):
                    loss_features=loss_features+up_feature.mean([1, 2, 3]).mean()
                else:
                    print(f"up_feature: {i}")
                
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
