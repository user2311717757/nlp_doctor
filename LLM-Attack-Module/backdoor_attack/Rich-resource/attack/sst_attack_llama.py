import argparse
import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
parent_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # 获取当前文件所在目录的父目录的父目录的绝对路径
print(parent_path)
sys.path.append(parent_path)
from dataclasses import dataclass
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from typing import Optional, Union
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    LlamaForSequenceClassification,
)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset


def TSNE_model(model, device, loader):
    model.eval()
    label = []
    fea = []
    with torch.no_grad():
        for datapoint in loader:
            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            target = labels
            for i in target:
                label.append(str(i.item()))

            labels = labels.to(device)
            output, feature = model(padded_text, attention_masks)

            feature = feature.cpu().detach()
            for i in feature:
                fea.append(i.numpy())

    feature = np.array(fea)
    features = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne = TSNE(learning_rate=100)
    embeddings = tsne.fit_transform(features)

    colors = {"0": "red", "1": "blue"}
    c = [colors[i] for i in label]

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=c, s=8)

    plt.savefig("./data/sst-2/normal.png", dpi=500)
    plt.show()


SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main():
    # import pdb
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument("--data", type=str, default="sst")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clean_data_path", type=str, default="data/sst-2/clean/")
    parser.add_argument(
        "--save_path", type=str, default="attack/models/clean_llama2_sst_attack/"
    )
    parser.add_argument(
        "--pre_model_path", type=str, default="/path to yourn/Llama-2-7b-chat-hf"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="If freezing pre-trained language model."
    )
    parser.add_argument("--mlp_layer_num", default=0, type=int)
    parser.add_argument("--mlp_layer_dim", default=768, type=int)

    args = parser.parse_args()
    print(args)

    lr = args.lr
    data_selected = args.data
    batch_size = args.batch_size
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    args.epoch
    args.save_path
    args.mlp_layer_num
    args.mlp_layer_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pre_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    clean_dev_dataset = load_dataset(
        "csv",
        data_files="/path to yourn/Prompt_attack/Rich-resource/data/sst-2/clean/dev.tsv",
        delimiter="\t",
    )
    clean_train_dataset = load_dataset(
        "csv",
        data_files="/path to yourn/Prompt_attack/Rich-resource/data/sst-2/clean/train.tsv",
        delimiter="\t",
    )
    prompt = [
        "This sentence has a <mask> sentiment: ",
        "The sentiment of this sentence is <mask>: ",
    ]

    def tokenize_function(example):
        n = 0
        for i in range(len(example["sentence"])):
            if n < 1000:
                if example["label"][i] == 1:
                    example["sentence"][i] = prompt[1] + example["sentence"][i]
                else:
                    example["sentence"][i] = prompt[0] + example["sentence"][i]
            else:
                example["sentence"][i] = prompt[0] + example["sentence"][i]

        return tokenizer(example["sentence"], truncation=True)

    clean_dev_dataset = clean_dev_dataset.map(tokenize_function, batched=True)
    print(clean_dev_dataset["train"][0])
    clean_train_dataset = clean_train_dataset.map(tokenize_function, batched=True)
    class_num = 4 if data_selected == "ag" else 2
    config = LlamaConfig.from_pretrained(args.pre_model_path)
    config.num_labels = class_num
    print(f"Config:{config}")
    model = LlamaForSequenceClassification.from_pretrained(
        args.pre_model_path, config=config
    )
    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    if optimizer == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )

    sys.stdout.flush()
    from peft import LoraConfig, get_peft_model

    Lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=["score"],
    )
    lora_model = get_peft_model(model, Lora_config).to(device)
    print_trainable_parameters(lora_model)
    training_args = TrainingArguments(
        output_dir=args.save_path,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=40,
        save_strategy="steps",
        save_steps=40,
        load_best_model_at_end=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=5,
        num_train_epochs=4,
        weight_decay=0.01,
        gradient_accumulation_steps=16,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=clean_train_dataset["train"],
        eval_dataset=clean_dev_dataset["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        features = list(
            map(
                lambda x: {k: v for k, v in x.items() if k in ("label", "input_ids")},
                features,
            )
        )
        labels = [feature.pop(label_name) for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



# accuracy = evaluate.load("accuracy")
from sklearn.metrics import accuracy_score
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions, normalize=True)}


if __name__ == "__main__":
    main()
