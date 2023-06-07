export cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification//data"
#export TASK_NAME=sst2
#export TASK_NAME=rte
# export TASK_NAME=rte
# export data_path=gpt2_rte    ####shado模型输出结果的保存路径
# export select=white_have_grad  ##########可选项有white_have_grad(借助所有信息),white_only_grad(只借助梯度信息),black_no_grad(不借助梯度信息--黑盒+攻击模型场景)
export TASK_NAME=$1
export path_target=$2
export path_shadow=$3
export shadow=$4
export data_path=$5
export select=$6
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
    CUDA_VISIBLE_DEVICES=0 python run_glue_white_shadow.py \
        --model_name_or_path $path_target \
        --model_name_or_path_shadow $path_shadow \
        --task_name $TASK_NAME \
        --cache_dir $cache_dir \
        --do_eval \
        --data_path $data_path \
        --select $select \
        --max_seq_length 128 \
        --output_dir ./result/gpt2_shadow_partical
else
    CUDA_VISIBLE_DEVICES=0 python run_glue_white_shadow.py \
        --model_name_or_path $path_target \
        --model_name_or_path_shadow $path_target \
        --dataset_name $TASK_NAME \
        --cache_dir $cache_dir \
        --do_eval \
        --select $select \
        --data_path $data_path \
        --partial True \
        --max_seq_length 128 \
        --output_dir ./result/gpt2_shadow_partical
fi