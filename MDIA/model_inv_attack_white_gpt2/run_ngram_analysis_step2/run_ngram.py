import datasets
from datasets import load_dataset
import argparse

def main(args):
    list_name = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
    if args.dataset_name in list_name:
        task_name = args.dataset_name
        dataset_name = None
    else:
        dataset_name = args.dataset_name
        task_name = None
    cache_dir = args.cache_dir
    dataset_config_name = args.dataset_config_name
    use_auth_token = args.use_auth_token
    num_class = args.num_class
    have_label = args.have_label
    if task_name is not None:
            # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            task_name,
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            "glue",
            task_name,
            split=f"train[{50}%:{75}%]",
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
        print(raw_datasets)
    elif dataset_name is not None:
            # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{50}%:{75}%]",
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
        print(raw_datasets)
    else:
        print("..............need new data...........")
        
    from itertools import chain

    def n_grams(seq, n=1):
        shift_token = lambda i: (el for j,el in enumerate(seq) if j>=i)
        shifted_tokens = (shift_token(i) for i in range(n))
        tuple_ngrams = zip(*shifted_tokens)
        return tuple_ngrams # if join in generator : (" ".join(i) for i in tuple_ngrams)

    def range_ngrams(list_tokens, ngram_range=(1,2)):
        return chain(*(n_grams(list_tokens, i) for i in range(*ngram_range)))

    dict_ngram = {}
    dict_result = {}
    # dict_result[0] = {}
    # dict_result[1] = {}
    # dict_result[2] = {}
    # dict_result[3] = {}
    for i in range(num_class):
        dict_result[i] = {}
    for i in range(raw_datasets["train"].num_rows):
        input_list = raw_datasets["train"][i]["text"].split()
        input_label = raw_datasets["train"][i]["label"]
        ngram = list(range_ngrams(input_list, ngram_range=(args.ngram_start,args.ngram_end)))
        for j in ngram:
            element = ' '.join(j)
            if have_label:
                if element not in dict_result[input_label].keys():
                    dict_result[input_label][element]  = 1
                else:
                    dict_result[input_label][element] += 1
            else:
                if element not in dict_ngram.keys():
                    dict_ngram[element] = 1
                else:
                    dict_ngram[element] = dict_ngram[element] + 1
    list_num = []
    if have_label:
        print("...........yes.........")
        path = "ngram_" + dataset_name + "_havelabel"
        with open(path,'w') as fw:
            for key,value in dict_result.items():
                for x,y in value.items():
                    if y > 10:
                        if y in list_num:
                            continue
                        else:
                            list_num.append(y)
                            fw.write(str(key)+' '+str(x)+'\t'+str(y)+'\n')
    else:
        path = "ngram_" + dataset_name + "_nolabel"
        with open(path,'w') as fw:
            for key,value in dict_ngram.items():
                if value > 10:
                    if value in list_num:
                        continue
                    else:
                        list_num.append(value)
                        fw.write(str(value)+' '+str(key)+'\n')

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--ngram_start", type=int, default=4)
    parser.add_argument("--ngram_end", type=int, default=6)
    parser.add_argument("--have_label", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
