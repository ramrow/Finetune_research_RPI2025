import argparse
import os
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType

os.environ["CUDA_VISIBLE_DEVICES"]="0"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def apply_chat_template(example):
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example['usr_prompt']},
        {"role": "assistant", "content": example["code_content"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt}

def tokenize_data(example):
    # tokens = tokenizer(example['text'], padding="longest",)
    tokens = tokenizer(example['text'], padding="max_length", max_length=1028, truncation=True)
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens


ds = (load_dataset("finalform/formated_foam",))
model="Qwen/Qwen-7B"
new_model = "qwen-foam"

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto",
    # bf16=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = "right"

#######################
tokenizer.chat_template =   "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}" \
                            "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ " \
                            "'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% if " \
                            "loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
#######################

organized_ds = ds.map(apply_chat_template)
tokenized_ds = organized_ds.map(tokenize_data)
tokenized_ds = tokenized_ds.remove_columns(["text", "system_prompt", "usr_prompt", "folder_name", "file_name", "case_path", "description", "code_content"])
