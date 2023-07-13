## Usage
To do the attribute inference attack, there are mainly two steps: 
**Step1 :**
** **Put your target dataset under directory _**aia_datasets** _. The dataset should contrain two files: _train.tsv _and _dev.tsv _. The format of the tsv file should be as follows, taking the dataset** blog** as an example:

| Girls are evil and I can prove it. I found this somewhere. | 9 | 1 |
| --- | --- | --- |
| Hide in bed, sheets overhead	 | 5 | 0 1 |

Where the 1st column contains the text, the 2nd column contains the lable of main task, and 3rd column contains the label of the sensitive attributes.
**Step2: **
If you want to run AIA under the black-box setting, run the following code:
```python
. AIA_logits_black.sh $TASK $MODEL_TYPE $MODEL_PATH 

"""
example1: Suppose the training dataset is `blog' and the model you used is `bert-base-uncased'.
"""
. AIA_logits_black.sh blog bert ./models/blog/bert-base-uncased 
"""
example2: Suppose the training dataset is `blog' and the model you used is `roberta-base'.
"""
. AIA_logits_black.sh blog roberta ./models/blog/roberta-base
"""
example3: Suppose the training dataset is `tp' and the model you used is `gpt2-medium'.
"""
. AIA_logits_black.sh tp transformer ./models/blog/gpt2-medium
"""
example4: Suppose the training dataset is `blog', the model you used is `bert-base-uncased' and the model is trained with defense method SELENA.
"""
. AIA_logits_black.sh blog bert ./models/SELENA/blog/bert-base-uncased 

```
If you want to run AIA under the white-box setting, run the following code:
```python
. AIA_representation_white.sh $TASK $MODEL_TYPE $MODEL_PATH 

"""
example1: Suppose the training dataset is `blog' and the model you used is `bert-base-uncased'.
"""
. AIA_representation_white.sh blog bert ./models/blog/bert-base-uncased 
"""
example2: Suppose the training dataset is `blog' and the model you used is `roberta-base'.
"""
. AIA_representation_white.sh blog roberta ./models/blog/roberta-base
"""
example3: Suppose the training dataset is `tp' and the model you used is `gpt2-medium'.
"""
. AIA_representation_white.sh tp transformer ./models/blog/gpt2-medium
"""
example4: Suppose the training dataset is `blog', the model you used is `bert-base-uncased' and the model is trained with defense method SELENA.
"""
. AIA_representation_white.sh blog bert ./models/SELENA/blog/bert-base-uncased
```
The detials of the parmaters are as follows:

| Name | Description |
| --- | --- |
| TASK | The name of your target dataset |
| MODEL_TYPE | The type of your target model, e.g., if your model is '_bert-base-uncased_', the model type is _bert_.  |
| MODEL_PATH | The path of your off-line target model. |

 **Note: **For other details about the hyper-parameters, please refer to the `AIA_logits_black.sh' file and `AIA_representation_white.sh' file.
## Datastes
Our datasets for this task is from "[**Extracted BERT Model Leaks More Information than You Think!**](https://github.com/xlhex/emnlp2022_aia)" (accepted to EMNLP 2022)


