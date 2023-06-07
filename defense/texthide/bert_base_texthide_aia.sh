#export TASK_NAME=sst2
#export TASK_NAME=qnli
#export TASK_NAME=ag_news
#export TASK_NAME=mrpc
#export TASK_NAME=ag_news
export aia_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/AIA/datasets/tp"
export target_or_shadow=target
#export path="/root/dataln/nianke/transformers/examples/pytorch/text-classification/result/bert_target_mrpc"
CUDA_VISIBLE_DEVICES=0,2,3 python run_glue_texthide_aia.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name $aia_dir \
  --cache_dir ../../data \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --target_or_shadow $target_or_shadow \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --output_dir ./aia/bert_target_tp
