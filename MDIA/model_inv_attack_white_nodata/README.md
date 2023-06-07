# Document Description
## Background
This folder contains the training scripts for MDIA(white-box/no data). 
## Useage

Description of the parameters in the script.

**model_name_or_path** : path of the target model.

**model_name_or_path_evaluation** : path of the evaluation model

**task_name/dataset_name** : name of data.If the data comes from a GLUE dataset you should use task_name, the rest should use datasets_name

**cache_dir** : cache directory to store downloaded data and models.

## bert_base_inversion_confidence.sh/roberta_base_inversion_confidence.sh/gpt2_inversion_confidence.sh
These three are scripts for MDIA under different models. The parameters in the corresponding scripts can be configured as described above. Then run
```
./bert_base_inversion_confidence.sh
```
Of course, the hyperparameters of the algorithm can also be modified, as shown in traner.py(line 2926).

## bert_base_dpsgd.sh/bert_base_kd.sh/bert_base_texthide.sh
These three scripts are for testing the effectiveness of the defences, you can run the corresponding defence scripts in the defence folder first. Then run
```
./bert_base_dpsgd.sh
```

## bert_base_inversion2mia.sh
This is the script for MDIA combined with MIA to generate data， can simply run
```
./bert_base_inversion2mia.sh
```
Of course, you can modify the number of generate data and save path in run_glue_inversion2mia.py(line 545 and 556)

