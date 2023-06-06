# Document Description
## Background
This folder contains all the runtime scripts for MIA (black-box/threshold).Specifically, this includes scripts for membership inference attacks. Scripts to test the effectiveness of defenses. As well as MEA/MDIA combined with MIA's scripts.
## Useage
Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_shadow** : path of the shadow model.

**task_name/dataset_name** : path of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

**partial** ： If you assume that the attacker has access to partial data, you can set this parameter to True.

### bert_base_black_shadow.sh/roberta_base_black_shadow.sh/gpt2_medium_black_shadow.sh
These three are the scripts of MIA under different models.

For example, you want to do an attack on a Bert model trained on MRPC.You can follow the description above to set the corresponding parameters than to run ./bert_base_black_shadow.sh mrpc

### bert_base_dpsgd.sh/bert_base_kd.sh/bert_base_texthide.sh
These three are the scripts of the test effectiveness of defenses under Bert models. The methods of defence mainly include DP-SGD,SELENA and Texthide.The content of the three scripts is basically the same, the main difference lies in the path of the target model and shadow model. The model needs to be obtained using the corresponding defense script.

For example, you want to do an test on a Bert model trained on MRPC.You can follow the description above to set the corresponding parameters than to run ./bert_base_dpsgd.sh mrpc

### bert_base_black_shadow.sh/bert_base_inversion2mia.sh
These three are the MEA/MDIA combined with MIA's scripts. For bert_base_black_shadow.sh, you only need modify the path of target model(this can be obtained from the training of MEA). For bert_base_black_shadow.sh, firstly, you need to run the script of MDIA,
Then, you can get the generated data and modify data path (3177 row in the trainer.py). Finally run ./bert_base_inversion2mia.sh ag_news
