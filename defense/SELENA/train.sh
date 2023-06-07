export TARGET_DATASET=$1
export TARGET_MODEL_NAME=bert-base-uncased
export EPOCHS=$2
export idx=$3
export SEED=42
export K=25
export L=10


VIC_OUTPUT_DIR=./log/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED
if [ ! -d $VIC_OUTPUT_DIR ];then
    mkdir -p $VIC_OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=2,3 nohup python run_glue.py \
  --model_name_or_path $TARGET_MODEL_NAME \
  --dataset_name ./sub_datasets/K25_L10/$TARGET_DATASET/sub_model_$idx\
  --cache_dir ./data \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --seed $SEED \
  --output_dir ./models/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED \
  --overwrite_output_dir \
  > ./log/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED/target.txt 2>&1 