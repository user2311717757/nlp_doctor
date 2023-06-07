export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification//data"
#export TASK_NAME=sst2
#export TASK_NAME=rte
#export TASK_NAME=mrpc
# export TASK_NAME=rte
# export data_path=bert_rte    ####shadow模型输出结果的保存路径,横杆前为模型名称，横杆后为数据集名称,可选项为模型名称_数据集名称_(partial or 不写)
# export select=white_have_grad  ##########可选项有white_have_grad(借助所有信息),white_only_grad(只借助梯度信息),black_no_grad(不借助梯度信息--黑盒+攻击模型场景)
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
export data_path=$5
export select=$6
#--dataset_name $TASK_NAME \
#--task_name $TASK_NAME \
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
  CUDA_VISIBLE_DEVICES=0 python run_glue_white_shadow.py \
    --model_name_or_path $path_target\
    --model_name_or_path_shadow $path_shadow \
    --task_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --data_path $data_path \
    --select $select \
    --max_seq_length 128 \
    --output_dir ./result/white_partical
else
  CUDA_VISIBLE_DEVICES=2 python run_glue_white_shadow.py \
    --model_name_or_path $path_target\
    --model_name_or_path_shadow $path_target \
    --task_name $TASK_NAME \
    --cache_dir $cache_dir \
    --do_eval \
    --data_path $data_path \
    --select $select \
    --partial True \
    --max_seq_length 128 \
    --output_dir ./result/white_partical
fi
