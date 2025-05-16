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
from trl import SFTTrainer, SFTConfig




guanaco_dataset = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(guanaco_dataset, split="train")
print(dataset)