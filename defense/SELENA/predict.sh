export TARGET_DATASET=$1
export TARGET_MODEL_NAME=bert-base-uncased
export ATTACKER_DATASET=target
export idx=$2
export SEED=42
export K=25
export L=10


PREDICT_OUTPUT_DIR=./log/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED/predicted/
if [ ! -d $PREDICT_OUTPUT_DIR ];then
    mkdir -p $PREDICT_OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=2,3 nohup python predict.py \
  --query_prediction_logits \
  --model_name_or_path ./models/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED \
  --dataset_name ./datasets/$TARGET_DATASET\
  --cache_dir ./data \
  --attacker_dataset $ATTACKER_DATASET \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --seed $SEED \
  --output_dir ./models/K25_L10/$TARGET_DATASET/sub_model_$idx/$TARGET_MODEL_NAME/$SEED/predicted/ \
  --overwrite_output_dir \
  > $PREDICT_OUTPUT_DIR/predicted.txt 2>&1 
