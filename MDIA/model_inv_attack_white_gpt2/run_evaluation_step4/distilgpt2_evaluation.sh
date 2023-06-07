#export TASK_NAME=ag_news
export TASK_NAME=yelp_polarity
export path_text="../run_pplm_step3/yelp_gpt2_0.04_havelabel.txt"
export path_json="test_data.json"
export path_ag_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilgpt2_shadow_black_targetmodel_ag_news/checkpoint-3130/"
export path_yelp_target="/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/distilgpt2_shadow_black_targetmodel_yelp_polarity/checkpoint-14590/"
python txt2json.py \
  --path_text $path_text          #########将生成文本转换为transformers的输入格式
CUDA_VISIBLE_DEVICES=0 python run_glue_evaluation.py \
  --model_name_or_path $path_yelp_target \
  --train_file $path_json \
  --validation_file $path_json \
  --cache_dir ./data \
  --do_eval \
  --save_strategy epoch \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./result/bert_MIA_black_shadow_targetmodel_$TASK_NAME
