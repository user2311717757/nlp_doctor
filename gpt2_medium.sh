#export TASK_NAME=sst2
#export TASK_NAME=rte
export TASK_NAME=mrpc
#export TASK_NAME=ag_news
#export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_mrpc/checkpoint-100"
CUDA_VISIBLE_DEVICES=1,2,3 python run_glue.py \
  --model_name_or_path gpt2-medium \
  --task_name $TASK_NAME \
  --cache_dir ./data \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./result/gpt2_$TASK_NAME
