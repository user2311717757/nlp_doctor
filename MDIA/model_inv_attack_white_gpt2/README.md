# Document Description
## Background
This folder contains the training scripts for MDIA(white-box/shadow data). Four steps are involved: fine-tuning the language model, n-gram analysis, text generation and attack evaluation

## fine-tuning the language model(GPT2)

Description of the parameters in the script(run_language.sh).

**model_name_or_path** : the path of GPT2.

**dataset_name** ： target data.

**per_device_train_batch_size** : training batch_size.

You can run:
```
cd run_language_model_step1
./run_language.sh
```

## n-gram analysis

Description of the parameters in the script(run_ngram_analysis.sh).

**dataset_name** ： target data.

**num_class** ： types of labels.

**ngram_start/ngram_end** : ngram analysis's start value and end value.

**have_label** : Whether to generate prompts based on labels. If set to false, the auxiliary data is unlabelled data.

You can run:
```
cd run_ngram_analysis_step2
./run_ngram_analysis.sh
```

## text generation

Description of the parameters in the script(run.py).

**class_label** : types of labels.

**cond_text** : prompt.

**pretrained_model_path** : the path of the fine-tuning language model in step1.

**model_name_or_path** : the path of target model.

**num_labels** : the number of labels.

You can run:
```
cd run_pplm_step3
python run.py
if you want to test the effect of defence:
python run_dp.py or run_kd.py or run_texthide.py
```
## attack evaluation

Description of the parameters in the script(distilbert_evaluation.sh).

**path_text** : text generated from previous step.

You can run:
```
cd run_evaluation_step4
./distilbert_evaluation.sh
```
