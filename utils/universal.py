# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pytorch_lightning.callbacks import ModelCheckpoint
import os

class UniversalCheckpoint(ModelCheckpoint):
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('universal checkpoint callback')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=False)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=None, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)
        parser.add_argument('--save_on_train_epoch_end', action='store_true', default=None)

        parser.add_argument('--load_ckpt_id',default=0, type=int)
        parser.add_argument('--load_ckpt_path', default='result/stablediffusion_distill_zh', type=str)
        parser.add_argument('--every_n_steps', default=10, type=int)
        parser.add_argument('--text_encoder', default='chinese_clip')  ## ## mul_clip  chinese_clip  mt5  alt_clip
        parser.add_argument('--text_encoder_path', default='clip_cn_vit-h-14.pt')
        parser.add_argument('--hybrid_training', action='store_true', default=True)
        parser.add_argument('--KD', action='store_true', default=True)
        parser.add_argument('--noise_offset', default=0.5, type=float)
        return parent_args

    def __init__(self, args):
        super().__init__(monitor=args.monitor,
                         save_top_k=args.save_top_k,
                         mode=args.mode,
                         every_n_train_steps=args.every_n_train_steps,
                         save_weights_only=args.save_weights_only,
                         filename=args.filename,
                         save_last=args.save_last,
                         every_n_epochs=args.every_n_steps,
                         save_on_train_epoch_end=args.save_on_train_epoch_end)

