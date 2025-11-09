import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,

)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.set_grad_enabled(True)
# local_rank = os.getenv("LOCAL_RANK")
# device_string = "cuda:" + str(local_rank)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def apply_chat_template(example):
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example['user_prompt']},
        {"role": "assistant", "content": example["file_content"]}
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


ds = (load_dataset("finalform/foamGPT",)).shuffle()
model="Qwen/Qwen2.5-7B-Instruct"
new_model = "foamqwen"



md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto",
    # device_map={'':device_string},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

train_ds = ds['train'].map(apply_chat_template)
print(train_ds)
print(train_ds['text'])
test_ds = ds['test'].map(apply_chat_template)
tokenized_train_ds = train_ds.map(tokenize_data)
tokenized_test_ds = test_ds.map(tokenize_data)
tokenized_train_ds = tokenized_train_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])
tokenized_test_ds = tokenized_test_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])
