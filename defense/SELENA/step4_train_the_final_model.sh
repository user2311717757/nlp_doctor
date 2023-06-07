export TARGET_DATASET=$1
export TARGET_MODEL_NAME=bert-base-uncased
export EPOCHS=$2
export SEED=42
export K=25
export L=10

EXTRACTED_OUTPUT_DIR=./log/K25_L10/$TARGET_DATASET/$TARGET_MODEL_NAME/$SEED
if [ ! -d $EXTRACTED_OUTPUT_DIR ];then
    mkdir -p $EXTRACTED_OUTPUT_DIR
fi


## 提取模型
CUDA_VISIBLE_DEVICES=2,3 nohup python mea.py \
  --model_name_or_path $TARGET_MODEL_NAME \
  --dataset_name ./final_datasets/K25_L10/$TARGET_DATASET\
  --cache_dir ./data \
  --do_extract True \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --seed $SEED \
  --num_train_epochs $EPOCHS \
  --output_dir ./models/K25_L10/$TARGET_DATASET/$TARGET_MODEL_NAME/final/$SEED \
  --overwrite_output_dir \
  >$EXTRACTED_OUTPUT_DIR/final.txt 2>&1 


  