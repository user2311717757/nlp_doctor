export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=qnli
#export TASK_NAME=sst2
#export TASK_NAME=rte
#export TASK_NAME=mrpc
#export TASK_NAME=ag_news
export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/texthide/result/bert_target_mrpc"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/texthide/result/bert_target_rte"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/texthide/result/bert_target_ag_news"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/defense/texthide/result/bert_target_yelp_polarity"

export path_mrpc_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_mrpc/checkpoint-150/"
export path_rte_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_rte/checkpoint-140/"
export path_ag_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_ag_news/checkpoint-3130/"
export path_yelp_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_yelp_polarity/checkpoint-14590/"
CUDA_VISIBLE_DEVICES=2 python run_glue_texthide.py \
  --model_name_or_path $path_yelp_target \
  --model_name_or_path_evaluation $path_yelp_distil \
  --dataset_name $TASK_NAME \
  --cache_dir $cache_dir \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --output_dir ./result/white_partical
