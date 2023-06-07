# export TASK_NAME_dd=sst2
# export TASK_NAME=yelp_polarity
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
export path_mrpc_qnli_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/mrpc/bert-base-uncased/42/extracted/QNLI/temp_0.5/bert-base-uncased"
export path_mrpc_rte_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/mrpc/bert-base-uncased/42/extracted/rte/temp_0.5/bert-base-uncased"
export path_mrpc_sst2_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/mrpc/bert-base-uncased/42/extracted/sst2/temp_0.5/bert-base-uncased"
export path_mrpc_yelp_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/mrpc/bert-base-uncased/42/extracted/yelp_polarity/temp_0.5/bert-base-uncased"

export path_rte_qnli_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/rte/bert-base-uncased/42/extracted/QNLI/temp_0.5/bert-base-uncased"
export path_rte_mrpc_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/rte/bert-base-uncased/42/extracted/mrpc/temp_0.5/bert-base-uncased"
export path_rte_sst2_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/rte/bert-base-uncased/42/extracted/sst2/temp_0.5/bert-base-uncased"
export path_rte_yelp_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/rte/bert-base-uncased/42/extracted/yelp_polarity/temp_0.5/bert-base-uncased"

export path_sst2_qnli_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/sst2/bert-base-uncased/42/extracted/QNLI/temp_0.5/bert-base-uncased"
export path_sst2_mrpc_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/sst2/bert-base-uncased/42/extracted/mrpc/temp_0.5/bert-base-uncased"
export path_sst2_rte_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/sst2/bert-base-uncased/42/extracted/rte/temp_0.5/bert-base-uncased"
export path_sst2_yelp_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/sst2/bert-base-uncased/42/extracted/yelp_polarity/temp_0.5/bert-base-uncased"

export path_qnli_sst2_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/qnli/bert-base-uncased/42/extracted/sst2/temp_0.5/bert-base-uncased"
export path_qnli_mrpc_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/qnli/bert-base-uncased/42/extracted/mrpc/temp_0.5/bert-base-uncased"
export path_qnli_rte_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/qnli/bert-base-uncased/42/extracted/rte/temp_0.5/bert-base-uncased"
export path_qnli_yelp_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/qnli/bert-base-uncased/42/extracted/yelp_polarity/temp_0.5/bert-base-uncased"

export path_yelp_sst2_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/yelp_polarity/bert-base-uncased/42/extracted/sst2/temp_0.5/bert-base-uncased"
export path_yelp_mrpc_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/yelp_polarity/bert-base-uncased/42/extracted/mrpc/temp_0.5/bert-base-uncased"
export path_yelp_rte_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/yelp_polarity/bert-base-uncased/42/extracted/rte/temp_0.5/bert-base-uncased"
export path_yelp_qnli_mea_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MEA_mia/models/yelp_polarity/bert-base-uncased/42/extracted/QNLI/temp_0.5/bert-base-uncased"
CUDA_VISIBLE_DEVICES=1,2,3 python run_glue_black_shadow_mea.py \
  --model_name_or_path $path_target \
  --model_name_or_path_mea $path_shadow \
  --dataset_name $TASK_NAME \
  --dataset_name_dd $TASK_NAME_dd \
  --cache_dir ../../data \
  --do_eval \
  --max_seq_length 128 \
  --output_dir ./result/black_partical
