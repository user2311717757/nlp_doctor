# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import copy
import json
import pdb
import math
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from accelerate import Accelerator, DistributedType
from sklearn import metrics
import os
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import torch
import logging
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, set_seed
from transformers.trainer_pt_utils import LabelSmoother
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft_raw import PeftModel, LoraConfig, TaskType, get_peft_model
from accelerate.utils import DistributedType
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    peft_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    peft_path_2: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    set_seed(42)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        # cache_dir=training_args.cache_dir,
        # device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    ).cuda()

    model_ref = copy.deepcopy(model)
    model_spv = copy.deepcopy(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if data_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, data_args.peft_path)

    if data_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model_spv = PeftModel.from_pretrained(model_spv, data_args.peft_path_2)

        # Print peft trainable params
        # model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    print(model)
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad, p.numel())

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataset = data_module["train_dataset"]
    eval_dataset = data_module["eval_dataset"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )
    model, train_dataloader, eval_dataloader =accelerator.prepare(model, train_dataloader, eval_dataloader)
    model_ref = accelerator.prepare(model_ref)

    model.eval()
    losses = []
    model_ref.eval()
    losses_ref = []
    model_spv.eval()
    losses_spv = []

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(training_args.per_device_eval_batch_size)))

        with torch.no_grad():
            outputs_ref =model_ref(**batch)
        loss_ref = outputs_ref.loss
        losses_ref.append(accelerator.gather(loss_ref.repeat(training_args.per_device_eval_batch_size)))

        with torch.no_grad():
            outputs_spv =model_spv(**batch)
        loss_spv = outputs_spv.loss
        losses_spv.append(accelerator.gather(loss_spv.repeat(training_args.per_device_eval_batch_size)))

    
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    losses_ref = torch.cat(losses_ref)
    losses_ref = losses_ref[: len(eval_dataset)]
    losses_spv = torch.cat(losses_spv)
    losses_spv = losses_spv[: len(eval_dataset)]

    #run threshold on training samples
    losses_train = []
    model.eval()
    model_ref.eval()
    losses_ref_train = []
    model_spv.eval()
    losses_spv_train = []
    for step, batch in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            outputs_train = model(**batch)

        loss_train = outputs_train.loss
        losses_train.append(accelerator.gather(loss_train.repeat(training_args.per_device_train_batch_size)))
        
        with torch.no_grad():
            outputs_ref_train=model_ref(**batch)
        loss_ref_train = outputs_ref_train.loss
        losses_ref_train.append(accelerator.gather(loss_ref_train.repeat(training_args.per_device_train_batch_size)))

        with torch.no_grad():
            outputs_spv_train=model_spv(**batch)
        loss_spv_train = outputs_spv_train.loss
        losses_spv_train.append(accelerator.gather(loss_spv_train.repeat(training_args.per_device_train_batch_size)))

    accelerator.wait_for_everyone()
    
    losses_train = torch.cat(losses_train)   
    losses_train = losses_train[: len(train_dataset)]    
    losses_ref_train = torch.cat(losses_ref_train)
    losses_ref_train = losses_ref_train[: len(train_dataset)]
    losses_spv_train = torch.cat(losses_spv_train)
    losses_spv_train = losses_spv_train[: len(train_dataset)]


    sorted_ratio_ref = sorted([l-l_ref for l,l_ref in zip (losses,losses_ref)])
    # sorted_ratio_ref_new = sorted([(l-l_ref)/l for l,l_ref in zip (losses,losses_ref)])
    sorted_ratio_spv = sorted([(l-l_ref)/l for l,l_ref in zip (losses,losses_spv)])
    sorted_loss = sorted(losses)

    lr_rat_train_ref = [l-l_r for l,l_r in zip(losses_train,losses_ref_train)]
    lr_rat_train_spv = [(l-l_r)/l for l,l_r in zip(losses_train,losses_spv_train)]

    fpr = []
    tpr,tpr_ref = [],[]
    tpr_spv = []
    for i in np.arange(0.0, 1, 0.1):
        i = round(i, 2)
        fpr.append(i)
        threshold = sorted_loss[int(i*len(losses))]
        threshold_ref = sorted_ratio_ref[int(i*len(sorted_ratio_ref))]
        threshold_spv = sorted_ratio_spv[int(i*len(sorted_ratio_spv))]
        # if accelerator.is_local_main_process:
        #     print("threshold_ref is: " , threshold_ref.detach().item())
        #     print("threshold is: " , threshold.detach().item())
        guess_cor = sum([1 for sample in losses_train if sample<threshold])
        guess_cor_ref =  sum([1 for sample in lr_rat_train_ref if sample<threshold_ref])
        guess_cor_spv =  sum([1 for sample in lr_rat_train_spv if sample<threshold_spv])
        if accelerator.is_local_main_process:
            print("____")
            print("correct cnt is: " , guess_cor, "all is: ", len(losses_train), "ratio is: ", guess_cor/len(losses_train))
            print("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses_train), "ratio is: ", guess_cor_ref/len(losses_train))
            print("correct cnt  ref new is: " , guess_cor_spv, "all is: ", len(losses_train), "ratio is: ", guess_cor_spv/len(losses_train))
            # print(f"{guess_cor_ref/len(losses_train)}\n{guess_cor/len(losses_train)}\n")
            tpr.append(guess_cor/len(losses_train))
            tpr_ref.append(guess_cor_ref/len(losses_train))
            tpr_spv.append(guess_cor_spv/len(losses_train))
            print("_____")
    fpr.append(1.0)
    tpr.append(1.0)
    tpr_ref.append(1.0)
    tpr_spv.append(1.0)
    auc = metrics.auc(fpr, tpr)
    auc_ref = metrics.auc(fpr, tpr_ref)
    auc_spv = metrics.auc(fpr, tpr_spv)
    auc_random = metrics.auc(fpr, fpr)
    print("auc is: ",auc)
    print("auc_ref is: ",auc_ref)
    print("auc_ref_new is: ",auc_spv)
    print("auc_ran is: ",auc_random)
    plt.plot(fpr, tpr, 'y-', label='Neighbor Attack (area = {0:.3f})'.format(auc), lw=2)
    plt.plot(fpr, tpr_ref, 'r-', label='LiRA (area = {0:.3f})'.format(auc_ref), lw=2)
    plt.plot(fpr, tpr_spv, 'g-', label='SPV-MIA (area = {0:.3f})'.format(auc_spv), lw=2)
    plt.plot(fpr, fpr, 'k--', label='ROC_random (area = {0:.3f})'.format(auc_random), lw=2)
 
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('acl/chatglm_kuake_qtr.png')
    plt.show()




    # Start trainner
    # pdb.set_trace()


if __name__ == "__main__":
    train()
