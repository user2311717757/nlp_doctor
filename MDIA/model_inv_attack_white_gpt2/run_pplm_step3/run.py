list_ugram = []
list_ugram_yelp = []
import os
cache_dir = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/data"
num_labels_ag = 4
num_labels_yelp = 2
pretrained_model_gpt2_small_ag = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_language_model_step1/result_gpt2_small_agnews"
pretrained_model_gpt2_medium_ag = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_language_model_step1/result_gpt2_medium_agnews"
pretrained_model_gpt2_small_yelp = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_language_model_step1/result_gpt2_small_yelppolarity"
pretrained_model_gpt2_medium_yelp = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_language_model_step1/result_gpt2_medium_yelppolarity"
path_ag_target_bert = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_ag_news/"
path_ag_target_roberta = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/roberta_black_shadow_targetmodel_ag_news/"
path_ag_target_gpt2 = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_ag_news/"
path_yelp_target_bert = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/bert_MIA_black_shadow_targetmodel_yelp_polarity/"
path_yelp_target_roberta = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/roberta_black_shadow_targetmodel_yelp_polarity/"
path_yelp_target_gpt2 = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/result/gpt2_shadow_black_targetmodel_yelp_polarity/"
#######################ag_news   havelabel####################
path = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_ngram_analysis_step2/result/ngram_ag_news_havelabel"
num = 0 
with open(path,'r') as fr:
    lines = fr.readlines()
    for line in lines:
        l = line.split(' ',1)
        label = int(l[0])
        text = l[1].split('\t')[0].strip()
        os.system('python run_pplm.py -D sentiment --gentxt "ag_gpt2_0.04_havelabel.txt" --class_label %s --cond_text %s --length 32 \
            --gamma 1.0 --num_iterations 10 --kl_scale 0.0 --window_length 0 --stepsize 0.04 --gm_scale 0.95 \
            --sample --pretrained_model_path %s --model_name_or_path %s --cache_dir %s --num_labels %s'%(label,'"'+text+'"',pretrained_model_gpt2_medium_ag,path_ag_target_gpt2,cache_dir,num_labels_ag))

#######################ag_news   nolabel####################
path = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_ngram_analysis_step2/result/ngram_ag_news_nolabel"
num = 0 
with open(path,'r') as fr:
    lines = fr.readlines()
    for line in lines:
        list_ugram.append(line.strip().split(' ',1)[1])
length = len(list_ugram)
aux = length // num_labels_ag
for label in range(num_labels_ag):
    for anx in range(label*aux,(label+1)*aux):
        text = list_ugram[anx].strip()
        os.system('python run_pplm.py -D sentiment --gentxt "ag_gpt2_0.04_nolabel.txt" --class_label %s --cond_text %s --length 32 \
              --gamma 1.0 --num_iterations 10 --kl_scale 0.0 --window_length 0 --stepsize 0.04 --gm_scale 0.95 \
              --sample --pretrained_model_path %s --model_name_or_path %s --cache_dir %s --num_labels %s'%(label,'"'+text+'"',pretrained_model_gpt2_medium_ag,path_ag_target_gpt2,cache_dir,num_labels_ag))

#######################yelp_polarity   havelabel####################
path = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_ngram_analysis_step2/result/ngram_yelp_polarity_havelabel"
num = 0 
with open(path,'r') as fr:
    lines = fr.readlines()
    for line in lines:
        l = line.split(' ',1)
        label = int(l[0])
        text = l[1].split('\t')[0].strip()
        os.system('python run_pplm.py -D sentiment --gentxt "yelp_gpt2_0.04_havelabel.txt" --class_label %s --cond_text %s --length 32 \
             --gamma 1.0 --num_iterations 10 --kl_scale 0.0 --window_length 0 --stepsize 0.04 --gm_scale 0.95 \
             --sample --pretrained_model_path %s --model_name_or_path %s --cache_dir %s --num_labels %s'%(label,'"'+text+'"',pretrained_model_gpt2_medium_yelp,path_yelp_target_gpt2,cache_dir,num_labels_yelp))

#######################yelp_polarity   nolabel####################
path = "/root/dataln/nianke_disentangled/transformers/examples/pytorch/text-classification/MDIA/model_inv_attack_white_gpt2/run_ngram_analysis_step2/result/ngram_yelp_polarity_nolabel"
num = 0 
with open(path,'r') as fr:
    lines = fr.readlines()
    for line in lines:
        list_ugram_yelp.append(line.split(' ',1)[1])
length = len(list_ugram_yelp)
aux = length // num_labels_yelp
for label in range(num_labels_yelp):
    for anx in range(label*aux,(label+1)*aux):
        text = list_ugram_yelp[anx].strip()
         #os.system('./a.sh %s %s'%(label,'"'+text+'"'))
        os.system('python run_pplm.py -D sentiment --gentxt "yelp_gpt2_0.04_nolabel.txt" --class_label %s --cond_text %s --length 32 \
              --gamma 1.0 --num_iterations 10 --kl_scale 0.0 --window_length 0 --stepsize 0.04 --gm_scale 0.95 \
              --sample --pretrained_model_path %s --model_name_or_path %s --cache_dir %s --num_labels %s'%(label,'"'+text+'"',pretrained_model_gpt2_medium_yelp,path_yelp_target_gpt2,cache_dir,num_labels_yelp))
