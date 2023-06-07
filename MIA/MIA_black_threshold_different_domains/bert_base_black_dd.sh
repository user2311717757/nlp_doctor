# export TASK_NAME_dd=sst2
# export TASK_NAME=qnli
export TASK_NAME=$1
export TASK_NAME_dd=$2
export path_target=$3
export path_shadow=$4
#export TASK_NAME=rte
#export TASK_NAME=ag_news
#export TASK_NAME=yelp_polarity
#export TASK_NAME=sst2
#export TASK_NAME=qnli
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_mrpc/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_rte/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/"
export path_sst2_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_sst2/"
export path_qnli_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_qnli/"
CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow.py \
  --model_name_or_path $path_target \
  --model_name_or_path_shadow $path_shadow \
  --task_name $TASK_NAME \
  --dataset_name_dd $TASK_NAME_dd \
  --cache_dir ../../data \
  --do_eval \
  --max_seq_length 128 \
  --output_dir ./result/black_partical
