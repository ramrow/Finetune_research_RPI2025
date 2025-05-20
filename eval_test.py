import torch
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

import numpy as np

# Step 1: Load the dataset
dataset = load_dataset("AdiOO7/llama-2-finance")

# Step 2: Load the tokenizer and model
base_model = "NousResearch/Llama-2-7b-chat-hf"

training_args = TrainingArguments("test_trainer")
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1
training_args.output_dir = "./eval_results"
training_args.per_device_eval_batch_size=1
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    peft_config = peft_params
    # compute_metrics=compute_metrics,
)

trainer.evaluate()