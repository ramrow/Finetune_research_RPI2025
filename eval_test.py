from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from trl import SFTConfig,SFTTrainer
from peft import LoraConfig
import numpy as np
import evaluate
import torch

dataset = load_dataset("AdiOO7/llama-2-finance", split="train")
metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_function(examples):
    t1 = []
    t2 = []
    for line in examples["text"]:
        temp = line.split(" ### Assistant: ")
        t1.append(temp[0])
        t2.append(temp[1])

    t = tokenizer(t1,padding="max_length", truncation=True)
    t["labels"] =  t2
    return t

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset.column_names, batched=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForSequenceClassification.from_pretrained("NousResearch/Llama-2-7b-chat-hf", quantization_config=quant_config, device_map={"": 0}
)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./eval_results",
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
    packing=False,
    max_seq_length=None,
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets,
    train_dataset=tokenized_datasets,
    compute_metrics=compute_metrics,
    peft_config=peft_params
)

trainer.evaluate()