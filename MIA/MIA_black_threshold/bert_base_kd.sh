export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=rte
#export TASK_NAME=mrpc
#export TASK_NAME=ag_news
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
#export TASK_NAME=sst2
#export TASK_NAME=qnli
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/mrpc/bert-base-uncased/42/final"
export path_mrpc_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/mrpc/shadow/bert-base-uncased/42/final"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/rte/bert-base-uncased/42/final"
export path_rte_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/rte/shadow/bert-base-uncased/42/final"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/ag_news/bert-base-uncased/42/final"
export path_ag_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/ag_news/shadow/bert-base-uncased/42/final"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/yelp_polarity/bert-base-uncased/42/final"
export path_yelp_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/kd/models/K25_L10/yelp_polarity/shadow/bert-base-uncased/42/final"

if [ $shadow -eq 1 ];then
  CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow.py \
  --model_name_or_path $path_target \
  --model_name_or_path_shadow $path_shadow \
  --task_name $TASK_NAME \
  --cache_dir $cache_dir \
  --do_eval \
  --max_seq_length 128 \
  --output_dir ./result/black_partical
else
  CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow.py \
    --model_name_or_path $path_target \
    --model_name_or_path_shadow $path_target \
    --dataset_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --partial True \
    --max_seq_length 128 \
    --output_dir ./result/black_partical
fi
