import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

def tokenize(examples):
        output = []
        texts = []
        for line in examples["text"]:
                temp = line.split("### ")
                output.append("### " + temp[3])
                texts.append("### " + temp[1] + "### " + temp[2])
        return tokenizer(texts,
                            text_target=output,
                            truncation=True,
                            padding="max_length",
                            max_length=5000,
                            return_tensors="pt")


model_name = "codellama/CodeLlama-7b-Python-hf"
data_name = "AdiOO7/llama-2-finance"
new_model = "llama-7b-Python-hf-finance-sentiment-recognition"

dataset = load_dataset(data_name, split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
print(len(tokenized_dataset['labels']), len(tokenized_dataset['attention_mask']), len(tokenized_dataset['input_ids']))
# tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")
# tokenized_dataset.set_format(type="torch", columns=["labels", "attention_mask"])

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
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
    # label_names=["labels"],  # Important for custom label columns
)

trainer = SFTTrainer(
    model=model,
    args=training_params,
    peft_config=peft_params,

    train_dataset=tokenized_dataset
)
trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# base_model = "codellama/CodeLlama-7b-Python-hf"

# test_dataset = "mlabonne/guanaco-llama2-1k"

# # new_model = "llama-7b-Python-hf-finance-sentiment-recognition"
# new_model = "llama-7b-Python-hf-test-guanaco"


# dataset = load_dataset(test_dataset, split="train")

# compute_dtype = getattr(torch, "float16")

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=False,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     quantization_config=quant_config,
#     device_map="auto"
# )
# model.config.use_cache = False
# model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# peft_params = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# training_params = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     optim="paged_adamw_32bit",
#     save_steps=25,
#     logging_steps=25,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     fp16=False,
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant",
#     report_to="tensorboard"
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_params,
#     # processing_class=tokenizer,
#     # dataset_text_field="text",
#     # max_seq_length=None,
#     # tokenizer=tokenizer,
#     args=training_params,
#     # packing=False,
# )

# trainer.train()
# trainer.model.save_pretrained(new_model)
# trainer.tokenizer.save_pretrained(new_model)