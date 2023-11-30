# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from tqdm import tqdm
import re
import shutil
import braceexpand
import json
import numpy as np

import webdataset as wds
from PIL import Image
import zhconv

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision.transforms.functional import crop
from torchvision.utils import save_image
from torchvision import transforms

from prefetch_generator import BackgroundGenerator


USED_KEYS = {"jpg": "instance_images", "json": "instance_prompt_ids"}
SIZE = 512

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def verify_keys(samples, required_keys, hr_size, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        if "json" not in sample:continue
        sample_json = sample["json"]
        if "watermark" in sample_json:
            watermark = sample_json["watermark"]
            aesthetic_score = sample_json["aesthetic_score"]
            if aesthetic_score<6 or watermark>0.5:
                continue
        yield {key:sample[key] for key in required_keys}


key_verifier = wds.filters.pipelinefilter(verify_keys)

def crop_left_upper(image):
    w,h = image.size
    if min(w,h)<SIZE:
        size = min(w,h)
    detla_w = w-size
    detla_h = h-size
    x = random.randint(0,detla_w)
    y = random.randint(0,detla_h)
    return (y,x),crop(image, y, x, size, size)
    

def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=[],
            hr_size=-1,
            size= SIZE,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=True
    ):
        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.center_crop = center_crop
        self.crop = transforms.CenterCrop(size) if center_crop else transforms.Lambda(crop_left_upper)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer = tokenizer

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, hr_size=hr_size, handler=handler))

        self.append(wds.map(self.preproc))

    def preproc(self, sample):
        """Applies the preprocessing for images"""


        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["original_size"] = instance_image.size

        # resize
        instance_image = transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
        # crop
        if self.center_crop:
            w,h = instance_image.size
            instance_image = self.crop(instance_image)
            if min(w,h)>SIZE:
                example["crops_coords_top_left"] = (int(h/2-256),int(w/2-256))
            else:
                example["crops_coords_top_left"] = (int(h/2-min(w,h)/2),int(w/2-min(w,h)/2))
        else:
            example["crops_coords_top_left"],instance_image = self.crop(instance_image)

        example["instance_images"] = self.image_transforms(instance_image)

        sample_json = sample["json"]


        if "caption_ori" in sample_json and str_contain_chinese(sample_json["caption_ori"]): ## wukong_dataset
            sample_json["caption_ori"] = zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；“”1234567890]', '', sample_json["caption_ori"]), 'zh-hans')
            example["instance_prompt_ids"] = sample_json["caption_ori"] 
            example["zh_or_not"] = 1
        elif "caption_ori_zh" in sample_json and "caption_ori" not in sample_json and str_contain_chinese(sample_json["caption_ori_zh"]): ## laion_zh_webdataset laion0.3B_trans_webdataset
            sample_json["caption_ori_zh"] = zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；“”1234567890]', '', sample_json["caption_ori_zh"]), 'zh-hans')
            example["instance_prompt_ids"] = sample_json["caption_ori_zh"] 
            example["zh_or_not"] = 1      
        elif "caption_ori_en" in sample_json and str_contain_chinese(sample_json["caption_ori_en"]): ## data_scraping
            sample_json["caption_ori_en"] = zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；“”1234567890]', '', sample_json["caption_ori_en"]), 'zh-hans')
            example["instance_prompt_ids"] = sample_json["caption_ori_en"] 
            example["zh_or_not"] = 1       
        else:
            if "caption_zh" in sample_json:
                example["instance_prompt_ids"] = sample_json["caption_zh"] 
                example["zh_or_not"] = 0   
            else:  ## 自爬中文数据caption为空
                example["instance_prompt_ids"] = ""
                example["zh_or_not"] = 0   

        if "caption_en" in sample_json:
            example["instance_en"] = sample_json["caption_en"]
        else:
            example["instance_en"] = ""

        example["input_ids"] = self.tokenizer([example["instance_prompt_ids"]], context_length=77)
        example["input_ids_uncond"] = self.tokenizer([""], context_length=77)

        return example

def collate_fn(examples):
    # print(examples)
    instance_prompt_ids = [example["instance_prompt_ids"] for example in examples]
    texts_en = [example["instance_en"] for example in examples]
    original_size = [example["original_size"] for example in examples]
    input_ids = [example["input_ids"] for example in examples]
    input_ids_uncond = [example["input_ids_uncond"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    crops_coords_top_left = [example["crops_coords_top_left"]for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    zh_or_not = [example["zh_or_not"] for example in examples]

    batch = {
        "pixel_values": pixel_values,
        "instance_prompt_ids": instance_prompt_ids,
        "original_size": torch.tensor(original_size),
        "crops_coords_top_left": torch.tensor(crops_coords_top_left),
        "input_ids": torch.cat(input_ids),
        "input_ids_uncond": torch.cat(input_ids_uncond),
        "texts_en":texts_en,
        "zh_or_not": torch.tensor(zh_or_not),

    }

    return batch


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train', default=False, action="store_true")
        parser.add_argument('--resample_train', default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str, default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument("--resolution", type=int, default=512)
        parser.add_argument("--center_crop", default=True)
        return parent_args

    def __init__(
        self,
        args,
        tokenizer,
        custom_collate_fn=None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer = tokenizer
        self.collate_fn = custom_collate_fn if custom_collate_fn is not None else collate_fn
        self.center_crop = args.center_crop
        self.resolution = args.resolution

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        self.train_prop = splits['train']
        self.train_dataloader = self._train_dataloader
        self.datasets['train'] = None

        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + num_val == len(all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)
        
    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )
            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)

    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )



if __name__ == '__main__':
    # import open_clip
    from transformers import T5Tokenizer

    # tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
    paths = "/data_share/data/multimodel_data/clip_model/mt5-xl"
    tokenizer = T5Tokenizer.from_pretrained(paths)

    url = "/public_data/ma/data_process_2023/laion2B_webdataset_2/{}.tar"
    available_shards = list(range(0, 10))
    urls = [url.format(str(shard).zfill(5)) for shard in available_shards]
    ds = ImageEmbeddingDataset(
                urls,
                tokenizer=tokenizer,
                shuffle_shards=True,
                resample=False,
                hr_size=512,
                handler=wds.handlers.warn_and_continue
            )
    # for item in iter(ds):
    #     print(item)
    #     break
    from prefetch_generator import BackgroundGenerator
    
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    
    loader = DataLoaderX(
            ds,
            num_workers=2,
            batch_size=4,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )
    f = open("caption.txt","w")
    for i,batch in enumerate(tqdm(loader)):
        if i<400:
            for t in batch["instance_prompt_ids"]:
                f.write(t+"\n")
        else:
            f.close()
            break
            
            
        
