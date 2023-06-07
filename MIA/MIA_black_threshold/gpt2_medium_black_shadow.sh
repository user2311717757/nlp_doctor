export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=sst2
#export TASK_NAME=rte
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
#export TASK_NAME=ag_news
#export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_mrpc/checkpoint-100/"
export path_mrpc_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_shadowmodel_mrpc/checkpoint-100/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_rte/checkpoint-105/"
export path_rte_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_shadowmodel_rte/checkpoint-105/"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_ag_news/checkpoint-1565/"
export path_ag_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_shadowmodel_ag_news/checkpoint-1565/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_yelp_polarity/checkpoint-7295/"
export path_yelp_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_shadowmodel_yelp_polarity/checkpoint-7295/"
if [ $shadow -eq 1 ];then
  CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow.py \
    --model_name_or_path $path_yelp_target \
    --model_name_or_path_shadow $path_yelp_shadow \
    --dataset_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --max_seq_length 128 \
    --output_dir ./result/gpt2_shadow_partical
else
  CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow.py \
    --model_name_or_path $path_mrpc_target \
    --model_name_or_path_shadow $path_mrpc_target \
    --task_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --partial True \
    --max_seq_length 128 \
    --output_dir ./result/gpt2_shadow_partical
fi
