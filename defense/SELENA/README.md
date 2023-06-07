## 1 Introduction
SELENA is a framework to train privacy-preserving models that induce similar behavior on member and non-member inputs to mitigate membership inference attacks (MIA).  It has two 
major components, namely Split-AI and Self-Distilation.  Split-AI, first splits the training data into random subsets, then trains a model on each subset of the data and finally get a new ensemble training sets through an adaptive inference strategy at test time. Self-Distillation transfers the knowledge of the model obtained by Split-AI into a new model by using the soft labels of the training set from Split-AI. 
## 2 Algorithm Details
The overall process of SELENA can be concluded as follows:

1. First, determine the total number of sub-models **K** : $F_1,F_2,...,F_K$ and the number** L** of sub-models which are not trained with it for each training sample in training set $D = \{(x_i,y_i)\}_{i=1}^n$; Then, generate the training sets for each sub-models based on the following rules:
> Initialize each sub-training sets $D_i$for each sub-models $F_i$to $\varnothing$;
> for $x_i$ in $D = \{(x_i,y_i)\}_{i=1}^n$do
> Randomly sample (K-L) indexes $Id_i$ from {1,2,..,K} and get the corresponding $Id_i^{\,non}$which represents the index set of the models whose training set doesn't include trainging sample $x_i$;
> for $k$ in $Id_i$do
> Append $x_i$ to $D_{k}$

2. Train each sub-models $F_i$with their own trainging sets $D_i$using SGD;
3. Generate the new labels $y'$for each samples in the traning sets $D$and get the privacy-preserving training set $D' = \{(x_i,y'_i)\}_{i=1}^n$;
> for $x_i$ in $D = \{(x_i,y_i)\}_{i=1}^n$do
> $y'_i = \frac{1}{L} \sum_{k \in Id_i^{\,non}} F_k(x_i)$

4. Train the final privacy-preserving model $F'$with training sets $D' = \{(x_i,y'_i)\}_{i=1}^n$using SGD;
## 3 How to run the code?
Taking SST-2 as an example. Suppose you want to train a** `**bert-base-uncased model' on `SST-2' using SELENA for 10 epochs, you should run the following files step by step:
```python
step1_split_datasets.ipynb
sh step2_1_train_sub-models.sh sst2 10
sh step2_2_batch_predict.sh sst2
python step3_average_label.py
sh step4_train_the_final_model.sh sst2 10
```
**Note:**  If you want to change the dataset, you just need to simply change the dataset path in the above files. Besides, if you want to change the model type, all you need do is to change the 	`TARGET_MODEL_NAME' in the `train. sh' and `step4_train_the_final_model' files. For other hyper-parameters details, please refer to the files.
