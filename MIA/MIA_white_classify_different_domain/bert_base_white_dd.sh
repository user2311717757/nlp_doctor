# export TASK_NAME_dd=yelp_polarity
# export TASK_NAME=mrpc
# export data_path=bert_mrpc    ####目标模型输出结果的保存路径,横杆前为模型名称，横杆后为数据集名称,可选项为模型名称_数据集名称_(partial or 不写)
# export data_path_dd=bert_yelp_polarity  ###########辅助模型输出结果保存路径
# export select=white_have_grad  ##########可选项有white_have_grad(借助所有信息),white_only_grad(只借助梯度信息),black_no_grad(不借助梯度信息--黑盒+攻击模型场景)
export TASK_NAME=$1
export TASK_NAME_dd=$2
export path_target=$3
export path_shadow=$4
export data_path=$5
export data_path_dd=$6
export select=white_have_grad
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
export path_mrpc_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_mrpc/checkpoint-100/"
export path_rte_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_rte/checkpoint-105/"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_ag_news/checkpoint-1565/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/checkpoint-7295/"
export path_qnli_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_qnli/checkpoint-1365/"
export path_sst2_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_sst2/checkpoint-880/"
CUDA_VISIBLE_DEVICES=1 python run_glue_white_shadow.py \
 --model_name_or_path $path_target \
 --model_name_or_path_shadow $path_shadow \
 --task_name $TASK_NAME \
 --dataset_name_dd $TASK_NAME_dd \
 --cache_dir ../data \
 --do_eval \
 --data_path $data_path \
 --data_path_dd $data_path_dd \
 --select $select \
 --max_seq_length 128 \
 --output_dir ./result/white_partical

