from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    task = "sst2"

    df_path = f"./datasets/{task}"
    df = pd.read_csv(df_path+"/train.tsv","\t")

    m1 = int(len(df) * 0.25)
    tmp_idx = list(range(m1))
    df = df.iloc[tmp_idx]


    def func(i):
        inference_dict = np.load(f"./sub_datasets/K25_L10/{task}/inference_dict.npy")
        tmp = []
        for j in inference_dict[i]:
            df_path = f"./predicted/{task}/sub_model_{j}/train.tsv"
            df = pd.read_csv(df_path,"\t")
            labels = [float(x) for x in df['label'].iloc[i].split()]
            tmp.append(labels)
        return " ".join(map(str,[round(x,5) for x in list(np.array(tmp).mean(axis=0))]))
    
    inference_dict = np.load(f"./sub_datasets/K25_L10/{task}/inference_dict.npy")
    n = len(inference_dict)
    
    with Pool(100) as p:
        average_predicts = list((tqdm(p.imap(func, range(n)), total=n, desc='Monitoring progress.')))

    print(f"Task finished.")
    
    df.label = average_predicts  
    if not os.path.exists(f"./final_datasets/K25_L10/{task}"):
        os.mkdir(f"./final_datasets/K25_L10/{task}")
    df.to_csv(f"./final_datasets/K25_L10/{task}/train.tsv","\t",index=0)

    dev_path = f"./datasets/{task}/dev.tsv"
    dev = pd.read_csv(dev_path,"\t")
    dev.to_csv(f"./final_datasets/K25_L10/{task}/dev.tsv","\t",index=0)