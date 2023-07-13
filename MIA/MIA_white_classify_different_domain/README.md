# Document Description
## Background
This folder contains all the experiments with different domain data for MIA (white-box/Classification Model).
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_shadow** : path of the shadow model.

**task_name/dataset_name** : path of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

**partial** ： If you assume that the attacker has access to partial data, you can set this parameter to True.

**data_path** ：The path to the output, before the horizontal bar, is the model name; after the horizontal bar, it is the dataset name.(optionally, the model name_dataset name_(partial or unwritten)).

**data_path_dd** ：The path to the output.

**select** : Options are white_have_grad (with all information), black_no_grad (without gradient information - black box + attack model scenario).

### bert_base_black_dd.sh

For example, you want to do an attack on a Bert model trained on MRPC (auxiliary data is RTE).
```
task_name = mrpc
task_name_dd = rte
target_model = result/bert_MIA_black_shadow_targetmodel_mrpc
shadow_model = result/bert_MIA_black_shadow_shadowmodel_rte
data_path = bert_mrpc
data_path_dd = bert_rte
./bert_base_black_dd.sh $task_name $task_name_dd $target_model $shadow_model $data_path $data_path_dd 
```
