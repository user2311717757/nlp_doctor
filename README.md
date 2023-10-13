# Privacy Evaluation Framework for NLP Models
## Descriptions
This repo contains source code and pre-processed corpora for "Privacy Evaluation Framework for NLP Models"

## Dependencies
transformers==4.26.0.dev0

torch==1.8.0+cu111

tokenizers==0.13.3

opacus==1.2.0

python==3.7.15

## Usage

Installing transformers can be found at https://github.com/huggingface/transformers. After installing it, You need to replace the four scripts in the package, trainer.py,modeling_bert.py,modeling_roberta.py,modeling_gpt2.py, with the scripts we provide.

In transformers, the addresses corresponding to these four scripts are：

trainer.py：transformers/src/transformers/trainer.py

modeling_bert.py: transformers/src/transformers/models/bert/modeling_bert.py

modeling_roberta.py: transformers/src/transformers/models/roberta/modeling_roberta.py

modeling_gpt2.py: transformers/src/transformers/models/gpt2/modeling_gpt2.py

Then, the text-classification folder(transformers/examples/pytorch/text-classification) in the transformer needs to be removed. For example, mv text-classification text-classification.bak. Next recreate a new text-classification folder and put in the remaining files we provide.

## How to train target model

You can use bert_base.sh,roberta_base.sh and gpt2_medium.sh to train target model.

```
***model_name_or_path** : path of the target model.

**task_name/dataset_name** : path of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

**target_or_shadow** : train target model or shadow model(can choise target or shadow)

run:
./bert_base.sh

If you can't download Bert model, you can download it manually from https://huggingface.co/bert-base-uncased/tree/main.
```

## attack and defense

We have placed the attack and defence scripts in five folders：MIA, MDIA, AIA, MEA, defense. The instructions for use can be found in the README.md in the corresponding folder.

