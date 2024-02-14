

###
python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \

python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 2e-5 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \


python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \

python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-4 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \



python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 2e-4 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \

python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 5e-4 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \



python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 8e-4 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \


python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 1e-3 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \


python  ../run_clm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir ./logs/  \
    --eval_steps 100 \
    --learning_rate 2e-3 \
    --block_size 1024 \
    --add_adapter \
    --adapter_reduction 2 \
    --do_ref_model \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \



