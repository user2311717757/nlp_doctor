# Document Description
## Background
This folder contains the training scripts for MDIA(white-box/no data) in auxiliary data from different domain. 
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_evaluation** : path of the evaluation model

**task_name/dataset_name** : name of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**dataset_name_dd** : name of different domain data.

**cache_dir** : cache directory to store downloaded data and models.

## bert_base_inversion_confidence.sh
You can run:
```
./bert_base_inversion_confidence.sh
```
Of course, the hyperparameters of the algorithm can also be modified, as shown in traner.py(line 2926).
