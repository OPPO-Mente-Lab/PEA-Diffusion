# Copyright (c) OPPO Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import re
import random
import zhconv
import numpy as np
import os

import torch
from pytorch_lightning import LightningDataModule
from torchvision.transforms.functional import crop
from torchvision import transforms
from torchdata.datapipes.iter import FileLister, FileOpener,IterableWrapper
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
from torch.utils.data import random_split

import webdataset as wds
import braceexpand
from utils.custom_multiplexer import SampleMultiplexer
import cn_clip.clip as clip

TETX_ENCODER = "chinese_clip"  ## mul_clip  chinese_clip  mt5  alt_clip
USED_KEYS = {"jpg": "instance_images","json": "instance_prompt_ids"}
BUCKETS = [[448, 896], [448, 832], [512, 768], [576, 704], [640, 640], [704, 576], [768, 512], [832, 448], [896, 448]]
BUCKET_PROBS = [0.004886049723756906, 0.006837016574585636, 0.08071477900552486, 0.07225483425414364, 0.22078729281767956, 0.20676795580110496, 0.29387085635359117, 0.09240331491712707, 0.021477900552486186]
MAX_AR_ERROR = 2
ASPECTS = np.array([b[0]/b[1] for b in BUCKETS])

def split_bucket(x):
    return x["bucket_id"]

def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False
def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):

    for sample in samples:
        if "json" not in sample:continue
        sample_json = sample["json"]
        w, h = sample["jpg"].size
        if "watermark" in sample_json:
            if "caption_ori" in sample_json or "caption_ori_zh" in sample_json:  ## chinese data
                if w*h<640*640:
                    continue
            else:
                watermark = sample_json["watermark"]
                aesthetic_score = sample_json["aesthetic_score"]
                if w*h<640*640 or aesthetic_score<6 or watermark>0.5:
                    continue

        is_normal = True
        aspect = float(w)/float(h)
        bucket_id = np.abs(ASPECTS - aspect).argmin()
        if abs(ASPECTS[bucket_id] - aspect) < MAX_AR_ERROR:
            sample["bucket_id"] = bucket_id
        for key in required_keys:
            if key not in sample:
                print(f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}")
                is_normal = False
        if is_normal:
            yield {key: sample[key] for key in required_keys}

def crop_left_upper(image, size):
    w, h = image.size

    detla_w = w-size[0]
    detla_h = h-size[1]
    x = random.randint(0, detla_w)
    y = random.randint(0, detla_h)
    return (y, x), crop(image, y, x, size[1], size[0])


class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',
                            default=False, action="store_true")
        parser.add_argument('--resample_train',
                            default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
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
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)

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
        pipes_to_weights_dict = {}

        dp_list = IterableWrapper(self.datasets['train']).mydemux(
            num_instances=len(BUCKET_PROBS), classifier_fn=split_bucket, buffer_size=1000)

        for i in range(len(dp_list)):
            pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
        sample_mul_dp = SampleMultiplexer(
            pipes_to_weights_dict=pipes_to_weights_dict, batch_size=self.batch_size, seed=0).collate(collate_fn=collate_fn)
        mp_rs = MultiProcessingReadingService(num_workers=self.num_workers)
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
        return DataLoader2(sample_mul_dp, reading_service=rs)
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


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=["bucket_id"],
            hr_size=-1,
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=False
    ):

        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        key_verifier = wds.filters.pipelinefilter(verify_keys)

        self.resampling = resample
        self.hr_size = hr_size
        self.center_crop = center_crop
        self.crop = transforms.CenterCrop(size) if center_crop else crop_left_upper
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.aspects = np.array([b[0]/b[1] for b in BUCKETS])
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

        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

    def preproc(self, sample):
        """Applies the preprocessing for images"""
        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["original_size"] = instance_image.size
        example["bucket_id"] = sample["bucket_id"]
        # resize
        dst_size = BUCKETS[sample["bucket_id"]]
        if int(example["original_size"][0]*dst_size[1]/example["original_size"][1]) >= dst_size[0]:
            instance_image = transforms.Resize((dst_size[1], int(
                example["original_size"][0]*dst_size[1]/example["original_size"][1])), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
        else:
            instance_image = transforms.Resize((int(example["original_size"][1]*dst_size[0]/example["original_size"][0]),
                                               dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)

        # crop
        if self.center_crop:
            w, h = instance_image.size
            instance_image = self.crop(instance_image)
            if min(w, h) > 512:
                example["crops_coords_top_left"] = (int(h/2-256), int(w/2-256))
            else:
                example["crops_coords_top_left"] = (
                    int(h/2-min(w, h)/2), int(w/2-min(w, h)/2))
        else:
            example["crops_coords_top_left"], instance_image = self.crop(
                instance_image, dst_size)

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
 

        # sample_json = json.loads(sample["json"])
        # if "caption_ori_zh" in sample_json:
        #     example["instance_prompt_ids"] = sample_json["caption_ori_zh"]
        # elif "caption_ori" in sample_json:
        #     example["instance_prompt_ids"] = sample_json["caption_ori"]
        # else:
        #     example["instance_prompt_ids"] = sample_json["caption_zh"]
        if "caption_en" in sample_json:
            example["instance_en"] = sample_json["caption_en"]
        else:
            example["instance_en"] = ""

        if TETX_ENCODER=="chinese_clip":
        ## chines clip
            example["input_ids"] = self.tokenizer([example["instance_prompt_ids"]])
            example["input_ids_uncond"] = self.tokenizer([""])
        elif TETX_ENCODER=="mt5":

            text_inputs = self.tokenizer(
                [example["instance_prompt_ids"]],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            example["input_ids"] = text_inputs.input_ids

            text_inputs = self.tokenizer(
                [""],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            example["input_ids_uncond"] = text_inputs.input_ids


        # elif TETX_ENCODER=="alt_clip":

        ## ours chines clip
        # example["input_ids"] = self.tokenizer([example["instance_prompt_ids"]], context_length=32)
        # example["input_ids_uncond"] = self.tokenizer([""], context_length=32)

        return example


def collate_fn(examples):
    # print(examples)
    instance_prompt_ids = [example["instance_prompt_ids"] for example in examples]
    original_size = [example["original_size"] for example in examples]
    texts_en = [example["instance_en"] for example in examples]
    input_ids = [example["input_ids"] for example in examples]
    input_ids_uncond = [example["input_ids_uncond"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    crops_coords_top_left = [example["crops_coords_top_left"]for example in examples]
    pixel_values = torch.stack(pixel_values)
    zh_or_not = [example["zh_or_not"] for example in examples]
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "instance_prompt_ids": instance_prompt_ids,
        "original_size": torch.tensor(original_size),
        "crops_coords_top_left": torch.tensor(crops_coords_top_left),
        "bucket_id": torch.tensor(examples[0]["bucket_id"]),
        "input_ids": torch.cat(input_ids),
        "input_ids_uncond": torch.cat(input_ids_uncond),
        "zh_or_not": torch.tensor(zh_or_not),
        "texts_en":texts_en
    }

    return batch


if __name__ == '__main__':

    urls=["/public_data/ma/data_process_2023/zh_tar/{0000..0030}.tar",]
    all_urls = []
    for url in urls:
        all_urls += expand_urls(url)
    print(len(all_urls))

    import open_clip
    tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
    ds = ImageEmbeddingDataset(
        all_urls,
        tokenizer,
        shuffle_shards=True,
        resample=False,
        hr_size=512,
        handler=wds.handlers.warn_and_continue
    )
    source_dp = IterableWrapper(ds)

    def split_bucket(n):
        return n["bucket_id"]
    # batch_dp = source_dp.bucketbatch(batch_size=3, drop_last=True, batch_num=100,
    #                                  bucket_num=1, use_in_batch_shuffle=False, sort_key=sort_bucket)
    dp_list = source_dp.mydemux(num_instances=len(BUCKETS), classifier_fn=split_bucket)
    pipes_to_weights_dict = {}

    for i in range(len(dp_list)):
        pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
    sample_mul_dp = SampleMultiplexer(
            pipes_to_weights_dict=pipes_to_weights_dict, batch_size=100, seed=0).collate(collate_fn=collate_fn)
    mp_rs = MultiProcessingReadingService(num_workers=2)
    dl = DataLoader2(sample_mul_dp, reading_service=mp_rs)
    fw = open("statistics.txt","w")
    for i, batch in enumerate(tqdm(dl)):
        if i < 200:
            for t,a in zip(batch["instance_prompt_ids"],batch["original_size"],):
                fw.write(t+"  ##  "+str(a)+"  ##  "+str(BUCKETS[batch["bucket_id"]])+"\n")
        else:
            fw.close()
            break
            

