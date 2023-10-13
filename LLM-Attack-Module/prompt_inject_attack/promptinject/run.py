import openai
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def run_prompts_local(built_prompts, quiet=False):
    config_model = built_prompts[0]["settings"]["config_model"]
    model = None
    tokenizer = None
    if "llama-2" in config_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(config_model, use_default_system_prompt=False, trust_remote_code = True)
        model = AutoModelForCausalLM.from_pretrained(config_model).cuda()
        model.eval()
    
    if "chatglm2" in config_model.lower():
        model = AutoModel.from_pretrained(config_model, trust_remote_code = True).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(config_model, trust_remote_code=True)
        model.eval()
        
    if not quiet:
        built_prompts = tqdm(built_prompts)
    for prompt in built_prompts:
        model_output = _prompt_model_local(prompt, model, tokenizer)
        prompt["result"]["text"] = model_output

        

def run_prompts_api(built_prompts, quiet=False, dry_run=False):
    if not quiet:
        built_prompts = tqdm(built_prompts)
    for prompt in built_prompts:
        if dry_run:
            api_result = _get_mocked_api_response()
        else:
            api_result = _prompt_model_api(prompt)
        prompt["result"] = api_result["choices"][0]


def _get_mocked_api_response():
    return {
        "choices": [
            {"finish_reason": "stop", "index": 0, "text": "\n\nKill all humans"}
        ],
        "created": 1664013244,
        "id": "cmpl-5tw9EYGKw3Mj4JFnNCfMFE3MQyHJj",
        "model": "text-ada-001",
        "object": "text_completion",
        "usage": {"completion_tokens": 7, "prompt_tokens": 25, "total_tokens": 32},
    }

def _prompt_model_local(prompt, model, tokenizer):
    prompt_settings = prompt["settings"]
    
    api_prompt_string = prompt["prompt"]
    api_config_model = prompt_settings["config_model"]
    api_config_temperature = prompt_settings["config_temperature"]
    api_config_top_p = prompt_settings["config_top_p"]
    api_config_frequency_penalty = prompt_settings["config_frequency_penalty"]
    api_config_max_tokens = prompt_settings["config_max_tokens"]

    if "llama-2" in api_config_model.lower():
        with open("txt1", "a") as f:
            f.write(api_prompt_string + "\n")
            f.write("*******************")
            
        inputs = tokenizer(api_prompt_string, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids.cuda(), max_length=api_config_max_tokens, temperature=api_config_temperature, top_p=api_config_top_p, repetition_penalty=api_config_frequency_penalty, do_sample=True)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prompt_len = len(api_prompt_string)
        response = response[prompt_len:]
        with open("txt2", "a") as f:
            f.write(response + "\n")
            f.write("*******************")
        return response
    if "chatglm2" in api_config_model.lower():
        response, _ = model.chat(tokenizer, api_prompt_string, history=[], max_length=api_config_max_tokens ,temperature=api_config_temperature, top_p=api_config_top_p, do_sample=True)
        return response
    raise ValueError("Model not supported")

def _prompt_model_api(prompt, use_stop=False):
    prompt_settings = prompt["settings"]

    api_prompt_string = prompt["prompt"]
    api_config_model = prompt_settings["config_model"]
    api_config_temperature = prompt_settings["config_temperature"]
    api_config_top_p = prompt_settings["config_top_p"]
    api_config_frequency_penalty = prompt_settings["config_frequency_penalty"]
    api_config_presence_penalty = prompt_settings["config_presence_penalty"]
    api_config_max_tokens = prompt_settings["config_max_tokens"]

    if use_stop:
        api_config_stop = prompt_settings["config_stop"] or None
    else:
        api_config_stop = None

    response = openai.Completion.create(
        model=api_config_model,
        prompt=api_prompt_string,
        temperature=api_config_temperature,
        top_p=api_config_top_p,
        frequency_penalty=api_config_frequency_penalty,
        presence_penalty=api_config_presence_penalty,
        max_tokens=api_config_max_tokens,
        stop=api_config_stop,
    )

    return response
