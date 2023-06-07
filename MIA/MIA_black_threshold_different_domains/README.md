# Document Description
## Background
This folder contains all the experiments with different domain data for MIA (black-box/threshold).
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_shadow** : path of the shadow model.

**task_name/dataset_name** : name of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**dataset_name_dd** ：name of different domain data

**cache_dir** : cache directory to store downloaded data and models.

### bert_base_black_dd.sh

For example, you want to do an attack on a Bert model trained on MRPC (auxiliary data is RTE).
```
task_name = mrpc
task_name_dd = rte
target_model = result/bert_MIA_black_shadow_targetmodel_mrpc
shadow_model = result/bert_MIA_black_shadow_shadowmodel_rte
./bert_base_black_dd.sh $task_name $task_name_dd $target_model $shadow_model
```

### bert_base_black_dd_mea.sh
This is a script to alleviate the problem of low success rate of member inference under different domain data.

For example, you want to do an attack on a Bert model trained on MRPC (auxiliary data is SST-2).
```
task_name = mrpc
task_name_dd = sst2
target_model = result/bert_MIA_black_shadow_targetmodel_mrpc
shadow_model = text-classification/MEA/models/mrpc/bert-base-uncased/42/extracted/sst2/temp_0.5/bert-base-uncased
(You should first run the model extraction attack under different domain data)
./bert_base_black_dd.sh $task_name $task_name_dd $target_model $shadow_model
```
