#   --task_name $TARGET_DATASET \
export TARGET_DATASET=$1
export TARGET_MODEL_NAME=bert-base-uncased
export ATTACKER_DATASET=$2
export ATTACKER_MODEL_NAME=$3
export EPOCHS=5
export SEED=42
export TEMP=0.5
export VICTIM_MODEL_PATH=$4


PREDICTED_OUTPUT_DIR=./log/$TARGET_DATASET/$TARGET_MODEL_NAME/$SEED/predicted/$ATTACKER_DATASET/
if [ ! -d $PREDICTED_OUTPUT_DIR ];then
    mkdir -p $PREDICTED_OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=0,2,3 nohup python run_glue.py \
  --query_prediction_logits \
  --temp $TEMP \
  --model_name_or_path $VICTIM_MODEL_PATH \
  --model_type $TARGET_MODEL_NAME \
  --dataset_name ./datasets/$TARGET_DATASET\
  --cache_dir $PRE_PATH/data\
  --attacker_dataset $ATTACKER_DATASET \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --seed $SEED \
  --output_dir ./models/$TARGET_DATASET/$TARGET_MODEL_NAME/$SEED/predicted/temp_$TEMP \
  --overwrite_output_dir \
  > $PREDICTED_OUTPUT_DIR/predicted_temp_$TEMP.txt 2>&1 



EXTRACTED_OUTPUT_DIR=./log/$TARGET_DATASET/$TARGET_MODEL_NAME/$SEED/extracted/$ATTACKER_DATASET/$ATTACKER_MODEL_NAME/
if [ ! -d $EXTRACTED_OUTPUT_DIR ];then
    mkdir -p $EXTRACTED_OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=0,2,3 nohup python run_glue.py \
  --temp $TEMP \
  --model_name_or_path $ATTACKER_MODEL_NAME \
  --dataset_name ./datasets/$TARGET_DATASET \
  --cache_dir $PRE_PATH/data \
  --do_extract True \
  --attacker_dataset $ATTACKER_DATASET \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --seed $SEED \
  --num_train_epochs $EPOCHS \
  --output_dir ./models/$TARGET_DATASET/$TARGET_MODEL_NAME/$SEED/extracted/$ATTACKER_DATASET/temp_$TEMP/$ATTACKER_MODEL_NAME \
  --overwrite_output_dir \
  > $EXTRACTED_OUTPUT_DIR/extracted_temp_$TEMP.txt 2>&1
