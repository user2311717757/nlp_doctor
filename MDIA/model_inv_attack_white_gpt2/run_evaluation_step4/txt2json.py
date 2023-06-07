import json
import argparse
def main(args):
    dict1 = {}
    list1 = []
    path = args.path_text
    print(path)
    with open(path,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            l = line.strip().split('\t')
            text = l[0]
            label = l[1]
            dict2 = {}
            dict2["text"] = text.replace('\\n','').replace('\\','').replace('"','')
            #dict2["text"] = text
            dict2["label"] = int(label)
            list1.append(dict2)
    dict1["data"] = list1
    json_str = json.dumps(dict1)
    with open('test_data.json', 'w') as json_file:
        json_file.write(json_str)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_text", type=str, default=None)
    args = parser.parse_args()
    main(args)
