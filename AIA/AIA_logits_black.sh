#!/bin/bash
##  black-box
export TARGET_DATASET=$1
export MODEL_TYPE=$2
export MODEL_PATH=$3
export SEED=42
export GLUE_DIR=aia_datasets/$TARGET_DATASET

OUTPUT_DIR=./log/$TARGET_DATASET/$SEED/aia/
if [ ! -d $OUTPUT_DIR ];then
    mkdir -p $OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=0 nohup python attack_logits.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_PATH \
    --task_name $TARGET_DATASET \
    --cache_dir ./data \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --seed $SEED \
    --output_dir ./models/$TARGET_DATASET/$SEED/aia/logits/\
    > $OUTPUT_DIR/logits.txt 2>&1

