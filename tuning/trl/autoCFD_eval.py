import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForVision2Seq ,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.set_grad_enabled(True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


ds = (load_dataset("YYgroup/NL2FOAM",)).shuffle()
model="YYgroup/AutoCFD-7B"

print(ds['train'])

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

def tokenize_data(example):
    inputs = tokenizer(example['description'], padding="max_length", truncation=True, return_tensors="pt")
    outputs = tokenizer(example['foamfiles'], padding="max_length", truncation=True, return_tensors="pt")
    tokens = {
        'input_ids': inputs['input_ids'],
        'labels': outputs['input_ids']
        }
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

train_ds = ds['train'].map(tokenize_data)
data = train_ds.remove_columns(['case_path', 'rel_path', 'mesh_path', 'description', 'mesh_content', 'foamfiles', 'file_tree', 'allrun', 'patch_names', 'instruction', 'input', 'output'])

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32, #change rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)
training_args = SFTConfig(
    output_dir="autocfdresult",
    # resume_from_checkpoint="./qwen_results/checkpoint-",
    # compute loss every few steps 1.5k/step
    num_train_epochs=7,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, #2
    optim="paged_adamw_32bit", #paged_adamw_32bit
    # save_steps=750,
    logging_steps=25,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
)

peft_md = get_peft_model(md, peft_params)

trainer = SFTTrainer(
    model=peft_md,
    train_dataset=data,
    eval_dataset=data,
    args=training_args,
    processing_class=tokenizer,
)

trainer.evaluate()