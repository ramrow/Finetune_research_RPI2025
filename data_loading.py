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
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import types

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

ds = load_dataset("processed_foam.json")
model="codellama/CodeLlama-7b-hf"
new_model = "llama-7b-foam"

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map={"": 0}
)
md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=3e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    packing=False,
    max_seq_length=None,
)

trainer = SFTTrainer(
    model=md,
    train_dataset=ds,
    peft_config=peft_params,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)