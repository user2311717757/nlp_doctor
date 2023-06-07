export path_medium="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data/models--gpt2-medium/snapshots/a8543b814a49bbee7dee6faa0a915c90f461f4d5"
export path_small="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
export data=ag_news
CUDA_VISIBLE_DEVICES=3,1,2,0 python run_clm.py \
  --model_name_or_path $path_small \
  --dataset_name $data \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --cache_dir /root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data \
  --do_train \
  --do_eval \
  --output_dir ./result_gpt2_small_ag_news

