# Document Description
## Background
This folder contains all the runtime scripts for MIA (black-box or white-box / Classification Model ).Specifically, this includes scripts for membership inference attacks, scripts to test the effectiveness of defenses, as well as MEA/MDIA combined with MIA's scripts.
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_shadow** : path of the shadow model.

**task_name/dataset_name** : path of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

**partial** ： If you assume that the attacker has access to partial data, you can set this parameter to True.

**data_path** ：The path to the output, before the horizontal bar is the model name, after the horizontal bar is the dataset name, optionally the model name_dataset name_(partial or unwritten).

**select** : Options are white_have_grad (with all information), black_no_grad (without gradient information - black box + attack model scenario).

### bert_base_white_shadow.sh/roberta_base_white_shadow.sh/gpt2_medium_white_shadow.sh

These three are the scripts of MIA under different models.

For example, you want to do an attack(auxiliary data is shadow data) on a Bert model trained on MRPC.
```
task_name = mrpc
target_model = result/bert_MIA_black_shadow_targetmodel_mrpc
shadow_model = result/bert_MIA_black_shadow_shadowmodel_mrpc
shadow = 1
data_path = bert_mrpc
select = white_have_grad
./bert_base_white_shadow.sh $task_name $target_model $shadow_model $shadow $data_path $select
```
if auxiliary data is partial data
```
task_name = mrpc
target_model = result/bert_MIA_black_shadow_targetmodel_mrpc
shadow_model = result/bert_MIA_black_shadow_shadowmodel_mrpc
shadow = 0
data_path = bert_mrpc
select = white_have_grad
./bert_base_white_shadow.sh $task_name $target_model $shadow_model $shadow $data_path $select
```

### bert_base_dpsgd.sh/bert_base_kd.sh/bert_base_texthide.sh
These three are the scripts of the test effectiveness of defenses under Bert models. The defence model needs to be obtained using the corresponding defense script.

For example, you want to do an attack(auxiliary data is shadow data) on a Bert model trained on MRPC.
```
task_name = mrpc
target_model = defense/DP-SGD/models/mrpc_dpsgd_targetmodel_5.0
shadow_model = defense/DP-SGD/models/mrpc_dpsgd_shadowmodel_5.0
shadow = 1
data_path = bert_mrpc
select = white_have_grad
./bert_base_dpsgd.sh $task_name $target_model $shadow_model $shadow $data_path $select
```

### bert_base_white_shadow.sh/bert_base_inversion2mia.sh
These three are the MEA/MDIA combined with MIA's scripts. For bert_base_black_shadow.sh, you only need modify the path of target model(this can be obtained from the training of MEA). 

For bert_base_black_shadow.sh, firstly, you need to run the script of MDIA,
Then, you can get the generated data and modify data path (3177 row in the trainer.py {MDIA/MDIA_white_nodata/inversion2mia_data/test_data_yelp_polarity.pt}). Finally run
```
task_name = yelp_polarity
target_model = result/bert_MIA_black_shadow_targetmodel_yelp_polarity
shadow_model = result/bert_MIA_black_shadow_targetmodel_yelp_polarity
shadow = 0
data_path = bert_mrpc
select = white_have_grad
./bert_base_dpsgd.sh $task_name $target_model $shadow_model $shadow $data_path $select
```
