#export TASK_NAME=ag_news
export TASK_NAME=yelp_polarity
export num_class=2   ##########有多少种类标签
export ngram_start=6
export ngram_end=8   #######ngram分析时ngram的取值
export have_label=True
#export TASK_NAME=yelp_polarity
python run_ngram.py \
  --dataset_name $TASK_NAME \
  --cache_dir ../../data \
  --have_label $have_label \
  --num_class $num_class \
  --ngram_start $ngram_start \
  --ngram_end $ngram_end \
