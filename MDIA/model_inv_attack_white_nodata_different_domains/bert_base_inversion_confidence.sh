export TASK_NAME_dd=qnli
#export TASK_NAME=qnli
#export TASK_NAME=rte
#export TASK_NAME=mrpc
export TASK_NAME=yelp_polarity
#export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_mrpc/checkpoint-100/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_rte/checkpoint-105/"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_ag_news/checkpoint-1565/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/checkpoint-7295/"
export path_qnli_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_qnli/checkpoint-1365/"
export path_sst2_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_sst2/checkpoint-880/"
export path_mrpc_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_mrpc/checkpoint-150/"
export path_rte_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_rte/checkpoint-140/"
export path_ag_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_ag_news/checkpoint-3130/"
export path_yelp_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_yelp_polarity/checkpoint-14590/"
export path_qnli_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_qnli/checkpoint-2730/"
export path_sst2_distil="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilbert_MIA_black_shadow_targetmodel_sst2/checkpoint-1760/"
CUDA_VISIBLE_DEVICES=0,1,2 python run_glue_inversion_confidence.py \
  --model_name_or_path $path_yelp_target \
  --model_name_or_path_evaluation $path_yelp_distil \
  --dataset_name $TASK_NAME \
  --dataset_name_dd $TASK_NAME_dd \
  --cache_dir ../data \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --output_dir ./result/white_partical
