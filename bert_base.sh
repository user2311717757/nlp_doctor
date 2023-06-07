#export TASK_NAME=sst2
#export TASK_NAME=qnli
#export TASK_NAME=rte
export TASK_NAME=mrpc
#export TASK_NAME=ag_news
#export TASK_NAME=yelp_polarity
export target_or_shadow=target
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/self_distil/models/K25_L10/mrpc/shadow/bert-base-uncased/42/final"
CUDA_VISIBLE_DEVICES=1,2,3 python run_glue.py \
  --model_name_or_path $path \
  --task_name $TASK_NAME \
  --target_or_shadow $target_or_shadow \
  --cache_dir ./data \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./result/bert_$TASK_NAME
