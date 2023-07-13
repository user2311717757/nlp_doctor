# Document Description
## Background
This folder contains the training scripts for Texthide. 
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model(bert-base-uncased).

**task_name/dataset_name** : path of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

**target_or_shadow** : Training a target model or a shadow model

## bert_base_texthide.sh/bert_base_texthide_aia.sh
```
./bert_base_texthide.sh
```
Because AIA has its own data, we are listing this separately.
