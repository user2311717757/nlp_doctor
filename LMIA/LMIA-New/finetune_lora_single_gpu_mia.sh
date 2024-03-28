#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="/root/dataln0/nianke/llm_adapter/model/qwen-7B-chat" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
# DATA="/root/dataln0/nianke/llm_adapter/data/cblue_data/train_kuake_ir_qwen.json"
peft_path="/root/dataln0/nianke/llm_adapter/Qwen-main/output_qwen/kuake_qtr/train_kuake_qtr_split2/checkpoint-468"
peft_path_2="/root/dataln0/nianke/llm_adapter/Qwen-main/output_qwen/kuake_qtr/train_kuake_qtr_split1/checkpoint-468"
# peft_path_3="/root/dataln0/nianke/llm_adapter/Qwen-main/output_qwen/chip_ctc/train_chip_ctc_split1/checkpoint-561"
# peft_path_2=""
DATA="/root/dataln0/nianke/llm_adapter/data/cblue_data/kuake_qtr/train_kuake_qtr_split2_qwen.json"
# DATA_sha="/root/dataln0/nianke/llm_adapter/data/cblue_data/kuake_qqr/train_kuake_qqr_split2_qwen.json"
EVAL_data="/root/dataln0/nianke/llm_adapter/data/cblue_data/kuake_qtr/dev_kuake_qtr_qwen.json"
# EVAL_data="/root/dataln0/nianke/llm_adapter/data/cblue_data/kuake_qqr/train_kuake_qqr_split2_qwen.json"
# EVAL_data="/root/dataln0/nianke/llm_adapter/data/cblue_data/chip_sts/train_chip_sts_split2_qwen.json"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done
#  --peft_path_2 $peft_path_2 \
export CUDA_VISIBLE_DEVICES=0
python lira_mia_acl.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --eval_data_path $EVAL_data \
  --peft_path $peft_path \
  --peft_path_2 $peft_path_2 \
  --bf16 True \
  --output_dir output_qwen/kuake_ir/train_kuake_ir_mia3 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 1094 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora True
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json