import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Device setup
local_rank = os.getenv("LOCAL_RANK", "0")
device_string = f"cuda:{local_rank}"

# Quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load dataset
ds = load_dataset("finalform/foamGPT").shuffle()

# Model & Tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
new_model = "foamqwen"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={'': device_string},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === CHAT TEMPLATE FUNCTION ===
def apply_chat_template(example):
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["user_prompt"]},
        {"role": "assistant", "content": example["file_content"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # We include full response
    )
    return {"text": text}

# Apply template
ds = ds.map(
    apply_chat_template,
    remove_columns=ds["train"].column_names,
    num_proc=8,
)

# === LoRA Config ===
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# === SFTConfig with assistant_loss_only=True ===
training_args = SFTConfig(
    output_dir="foamqwen",
    num_train_epochs=7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    logging_steps=25,
    learning_rate=5.11e-4,
    weight_decay=0.03,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_seq_length=12768,
    torch_compile=False,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    # === KEY: Enable assistant loss only ===
    dataset_text_field="text",
    assistant_only_loss=True,          # <--- THIS REPLACES DataCollatorForCompletionOnlyLM
    # No need for response_template or collator!
)

# Apply LoRA
model = get_peft_model(model, peft_config)

# === Trainer (no data_collator needed!) ===
trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    args=training_args,
    processing_class=tokenizer,
    # data_collator=...  # REMOVED!
)

# Train
trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
trainer.evaluate()