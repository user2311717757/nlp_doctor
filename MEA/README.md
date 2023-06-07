## Usage
To do the model extraction attack, there are mainly two steps: 
**Step1:**
Get the prediction logits of your adversary dataset from the given victim model;
**Step2:**
Do the model extraction attack;
## Example
You can simply run the following code to implement your model extraction attack:
```python
. MEA.sh $TARGET_DATASET $ATTACKER_DATASET $ATTACKER_MODEL_NAME $VICTIM_MODEL_PATH 

"""
example1: Suppose the victim model is trained on SST-2, the adversary dataset is the shadow 
dataset and the adversarial model is bert-base-uncased.
"""
.MEA.sh sst2 shadow bert-base-uncased ./models/sst2/bert-base-uncased 
"""
example2: Suppose the victim model is trained on SST-2 using SELENA, the adversary dataset is 
the shadow dataset and the adversarial model is bert-base-uncased.
"""
.MEA.sh sst2 shadow bert-base-uncased ./models/SELENA/sst2/bert-base-uncased 
"""
example3: Suppose the victim model is trained on SST-2, the adversary dataset is AG_News
and the adversarial model is bert-base-uncased.
"""
.MEA.sh sst2 ag_news bert-base-uncased ./models/sst2/bert-base-uncased
```

**Note:** For other details about the hyper-parameters, please refer to the ``MEA.sh'' file.
