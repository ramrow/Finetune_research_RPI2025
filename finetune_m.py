import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import json

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

def apply_format(example):
    messages = [
        {"role": "user", "content": example['text']},
        {"role": "assistant", "content": example['0/nuTilda']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

def tokenize_data(example):
    tokens = tokenizer(example['prompt'], padding="max_length", max_length=512)
    # Set padding token labels to -100 to ignore them in loss calculation
    print(tokens)
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

# def format_data(example):
#     prompt = example["text"]
#     response = f'[/INST] {example['0/nuTilda']}'
#     return {"text": prompt, "labels": response, "input_ids": ""}

# def tokenize_data(example):
#     prompt = example["text"]
#     output = example['labels']
#     example['input_ids'] = tokenizer(prompt, padding="max_length", max_length=512).input_ids
#     example['labels'] =  tokenizer(output, padding="max_length", max_length=512).input_ids

#     return example    

# ds = (load_dataset("finalform/processed_foam", split="train")).map(format_data)
ds = load_dataset("finalform/processed_foam", split="train")
model="NousResearch/Llama-2-13b-hf"
new_model = "llama-foam"

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto"
)
md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

organized_ds = ds.map(apply_format)
tokenized_ds = organized_ds.map(tokenize_data)
tokenized_ds = tokenized_ds.remove_columns(['prompt','text', 'allrun', '0/U', 'constant/transportProperties', 'constant/turbulenceProperties', '0/s', '0/sigma', 'constant/fvOptions', '0/omega', 'constant/MRFProperties', '0/k', 'system/fvSchemes', '0/nut', '0/p', '0/epsilon', 'system/controlDict', 'system/fvSolution', 'constant/dynamicMeshDict', '0/nuTilda', 'system/topoSetDict'])


peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# peft_md = get_peft_model(md, peft_params)

training_args = SFTConfig(
    output_dir="./llama_results_tildaONLY",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    packing=False,
)

trainer = SFTTrainer(
    model=md,
    train_dataset=tokenized_ds,
    peft_config=peft_params,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)