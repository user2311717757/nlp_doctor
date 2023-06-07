export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=rte
#export TASK_NAME=mrpc
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
#export TASK_NAME=sst2
#export TASK_NAME=qnli
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_mrpc/"
export path_mrpc_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_mrpc/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_rte/"
export path_rte_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_rte/"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_ag_news/checkpoint-1565/"
export path_ag_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_ag_news/checkpoint-1565/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/"
export path_yelp_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_yelp_polarity/checkpoint-7295/"
export path_qnli_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_qnli/checkpoint-1365/"
export path_qnli_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_qnli/checkpoint-1365/"
export path_sst2_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_sst2/"
export path_sst2_shadow="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_shadowmodel_sst2/"

# if [ $shadow -eq 1 ];then
#   CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_inversion2mia.py \
#   --model_name_or_path $path_target \
#   --model_name_or_path_shadow $path_target \
#   --task_name $TASK_NAME \
#   --cache_dir $cache_dir \
#   --do_eval \
#   --max_seq_length 128 \
#   --output_dir ./result/black_partical
# else
if [ $shadow -eq 0 ];then
  CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_inversion2mia.py \
    --model_name_or_path $path_target \
    --model_name_or_path_shadow $path_target \
    --task_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --partial True \
    --max_seq_length 128 \
    --output_dir ./result/black_partical
fi
