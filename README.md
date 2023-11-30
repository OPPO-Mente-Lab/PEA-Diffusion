# PEA-Diffusion


The official code for the paper [PEA-Diffusion: Parameter-Efficient Adapter with Knowledge Distillation in non-English Text-to-Image Generation](https://arxiv.org/abs/2311.17086).

## Requirements
A suitable [conda](https://conda.io/) environment named `PEA-Diffusion` can be created
and activated with:

```
conda create -n PEA-Diffusion   
source activate PEA-Diffusion   
pip install -r requirements.txt
```

## Data Prepare 
The English data we trained directly used [LAION](https://huggingface.co/datasets/laion/laion2B-en), and the Chinese data came from [WuKong](https://wukong-dataset.github.io/wukong-dataset/), [LAION_ZH](https://huggingface.co/datasets/IDEA-CCNL/laion2B-multi-chinese-subset). Our training data is [webdataset](https://github.com/webdataset/webdataset) format. If only multilingual-CLIP training is required, then only need English image-text pairs.
If training an PEA for a language-specific and aiming to generate images that are culturally relevant to that language, parallel corpora are necessary. We suggest download the [data](https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en) or translating the English prompts into the specific language. For more details, please refer to our paper.


## Training based SDXL

```bash
bash train_sdxl_zh.sh 0 8
```
The first parameter represents the global rank of the current process, used for inter process communication. The host with rank=0 is the master node.
and the second parameter is the world size. Please review the detailed parameters of model training with train_sdxl_zh.sh.

The training code includes a large number of model paths that need to be downloaded by yourself. For detailed download paths, please see Appendix 6.2 of the paper.

train_sdxl.py can run T21 in four other languages: Italian, Russian, Korean, and Japanese. Similarly, you need to download the corresponding clip model and put it in the corresponding path.

## Inference

We provide a script to generate images using pretrained checkpoints. run
```bash
python tests/test_sdxl_zh.py
```
For more downstream test scripts, please view the tests directory


## TODOs

- [x] Release inference code
- [x] Release training code
- [ ] Release PEA checkpoint
- [ ] Release demo


## Acknowledgements
we borrow some code from [TorchData](https://github.com/pytorch/data/blob/a5b4720dece60565788ac4c9a85e01719188b28e/torchdata/datapipes/iter/util/samplemultiplexer.py)


# Citation
```
@misc{ma2023peadiffusion,
      title={PEA-Diffusion: Parameter-Efficient Adapter with Knowledge Distillation in non-English Text-to-Image Generation}, 
      author={Jian Ma and Chen Chen and Qingsong Xie and Haonan Lu},
      year={2023},
      eprint={2311.17086},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
