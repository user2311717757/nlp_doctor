cache_dir="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
#export TASK_NAME=ag_news
export TASK_NAME=yelp_polarity
#export TASK_NAME=ag_news
#export TASK_NAME=mrpc
export target_or_shadow=shadow  
export EPOCHS=7
export EPSILON=15
export MAX_PHYSICAL_BATCH_SIZE=32
export MAX_GRAD_NORM=0.1
export LR=2e-4
export DELTA=1e-5
export output_dp_dir=./models/yelp_polarity_dpsgd_shadowmodel_15
CUDA_VISIBLE_DEVICES=0 python run_glue_dpsgd.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name $TASK_NAME \
  --cache_dir $cache_dir \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --target_or_shadow $target_or_shadow \
  --EPOCHS $EPOCHS \
  --LR $LR \
  --DELTA $DELTA \
  --EPSILON $EPSILON \
  --MAX_PHYSICAL_BATCH_SIZE $MAX_PHYSICAL_BATCH_SIZE \
  --MAX_GRAD_NORM $MAX_GRAD_NORM \
  --output_dp_dir $output_dp_dir \
  --output_dir ./result/bert_target_$TASK_NAME
