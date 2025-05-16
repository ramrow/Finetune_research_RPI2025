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
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer



MODEL_ID = "codellama/CodeLlama-7b-Python-hf"
DATASET_NAME  = "AdiOO7/llama-2-finance"
NEW_ID = "llama-7b-Python-hf-finance-sentiment-recognition"

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BF16 = True


LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_RANK = 64
TARGET_MODULES = [
    "q_proj", "v_proj"
]




bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Configure model parameters
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

    # Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)

def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # Split into train and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=970, seed=42
    )

    # Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_RANK,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
peft_model = get_peft_model(model, peft_config)



    # Define training arguments
training_args = TrainingArguments(
        output_dir="./result",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=25,
        save_steps=25,
        save_total_limit=3,
        warmup_ratio=0.03,
        fp16=True,
        bf16=BF16,
        seed=42,
        gradient_checkpointing=True,
    )

trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        peft_config=peft_config,
        data_collator=data_collator,
    )



trainer.train()
trainer.model.save_pretrained(NEW_ID)
trainer.tokenizer.save_pretrained(NEW_ID)




# dataset = load_dataset(data_name, split="train")
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
# # print(len(tokenized_dataset['labels']), len(tokenized_dataset['attention_mask']), len(tokenized_dataset['input_ids']))
# # tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")
# # tokenized_dataset.set_format(type="torch", columns=["labels", "attention_mask"])

# compute_dtype = getattr(torch, "float16")
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=False,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quant_config,
#     device_map={"": 0}
# )
# model.config.use_cache = False
# model.config.pretraining_tp = 1

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
#     report_to="tensorboard",
#     # label_names=["labels"]  # Important for custom label columns
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_params,
#     peft_config=peft_params,
#     processing_class=tokenizer,
#     train_dataset=dataset
# )


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