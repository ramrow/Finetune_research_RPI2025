from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

md, tk = FastLanguageModel(
    model_name= "unsloth/Qwen3-8B",
    max_seq_length= 1028,
    load_in_4bit=True,
)

md = FastLanguageModel.get_peft_model(
    model= md,
    r= 32,
    target_modules= "all-linear",
    bias= "none",
    lora_dropout= 0.1,
    lora_alpha=32, #could be 16
)

ds = load_dataset("finalform/split_foam",)
