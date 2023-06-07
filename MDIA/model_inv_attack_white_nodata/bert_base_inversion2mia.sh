#export TASK_NAME=qnli
#export TASK_NAME=sst2
#export TASK_NAME=rte
#export TASK_NAME=mrpc
export TASK_NAME=yelp_polarity
#export TASK_NAME=yelp_polarity
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_mrpc/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_rte/"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_ag_news/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/"
CUDA_VISIBLE_DEVICES=2 python run_glue_inversion2mia.py \
  --model_name_or_path $path_yelp_target \
  --dataset_name $TASK_NAME \
  --cache_dir ../../data \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --output_dir ./result/white_partical
