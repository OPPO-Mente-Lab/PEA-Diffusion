#!/bin/bash

export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
ROOT_DIR=./result  # 

MODEL_NAME=stablediffusion_distill_zh

MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=3
GPUS_PER_NODE=$2
MICRO_BATCH_SIZE=10

CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End
###         data/{00000..10315}.tar \
##         data/laion_zh_webdataset/{00000..14339}.tar \


DATA_ARGS="\
        --webdataset_base_urls \
        data/laion2B_webdataset/{00000..90000}.tar \
        data/BLIP_tar_512/{00000..11377}.tar \
        data/laion0.3B_trans_webdataset/{00000..31487}.tar \
        data/laion_400m/{00000..41511}.tar \
        data/laion2b/{00000..99999}.tar \
        data/coyo1/{00000..14016}.tar \
        data/coyo2/{00000..30000}.tar \
        data/data_scraping_theme/{00000..04770}.tar \
        data/data_scraping_2023/{00000..01023}.tar \
        data/data_scraping_0/{00200..04515}.tar \
        data/data_scraping_1/{00000..03637}.tar \
        data/aesthetics_tar_5/{00000..44487}.tar \
        data/{00000..10315}.tar \
        data/laion_zh_webdataset/{00000..14339}.tar \
        data/zh_tar/{0000..0035}.tar \
        data/zh_ta1/{0000..0016}.tar \
        --num_workers 2 \
        --batch_size $MICRO_BATCH_SIZE \
        --shard_width 5 \
        --hr_size 512 \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        --resample_train \
        "

MODEL_ARGS="\
        --model_path stable-diffusion-xl-protovisionXL \
        --text_encoder chinese_clip \
        --text_encoder_path clip_cn_vit-h-14.pt \
        --learning_rate 1e-5 \
        --weight_decay 1e-1 \
        --warmup_steps 100 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_id 0 \
        --load_ckpt_path result/stablediffusion_distill_zh \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --accelerator gpu \
        --devices $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
# echo $options
python print_args.py --model_args="$options"
# python finetune.py $options

export CC=gcc
export CXX=g++
python -m torch.distributed.run \
    --nnodes $NNODES \
    --master_addr 10.25.193.87 \
    --master_port 28890 \
    --node_rank $1 \
    --nproc_per_node $GPUS_PER_NODE \
    train_sdxl_zh.py $options

