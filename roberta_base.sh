#export TASK_NAME=emotion
#export TASK_NAME=rte
export TASK_NAME=mrpc
#export TASK_NAME=ag_news
#export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
CUDA_VISIBLE_DEVICES=0,2,3 python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --cache_dir ./data \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 15 \
  --output_dir ./result/roberta_black_shadow_targetmodel$TASK_NAME
