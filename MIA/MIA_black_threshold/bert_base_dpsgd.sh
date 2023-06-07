export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=rte
#export TASK_NAME=ag_news
#export TASK_NAME=ag_news
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
#export TASK_NAME=sst2
#export TASK_NAME=qnli
export path_ag_target_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/ag_news_dpsgd_targetmodel_15"
export path_ag_shadow_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/ag_news_dpsgd_shadowmodel_15"
export path_rte_target_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/rte_dpsgd_targetmodel_15"
export path_rte_shadow_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/rte_dpsgd_shadowmodel_15"
export path_mrpc_target_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/mrpc_dpsgd_targetmodel_15"
export path_mrpc_shadow_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/mrpc_dpsgd_shadowmodel_15"
export path_yelp_target_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/yelp_polarity_dpsgd_targetmodel_15"
export path_yelp_shadow_dpsgd_15="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/yelp_polarity_dpsgd_shadowmodel_15"

export path_ag_target_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/ag_news_dpsgd_targetmodel_5.0"
export path_ag_shadow_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/ag_news_dpsgd_shadowmodel_5.0"
export path_rte_target_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/rte_dpsgd_targetmodel_5.0"
export path_rte_shadow_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/rte_dpsgd_shadowmodel_5.0"
export path_mrpc_target_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/mrpc_dpsgd_targetmodel_5.0"
export path_mrpc_shadow_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/mrpc_dpsgd_shadowmodel_5.0"
export path_yelp_target_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/yelp_polarity_dpsgd_targetmodel_5.0"
export path_yelp_shadow_dpsgd_5="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/DP-SGD/models/yelp_polarity_dpsgd_shadowmodel_5.0"

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
  CUDA_VISIBLE_DEVICES=0,2,3 python run_glue_black_shadow.py \
    --model_name_or_path $path_target \
    --model_name_or_path_shadow $path_target \
    --task_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --partial True \
    --max_seq_length 128 \
    --output_dir ./result/black_partical
fi
