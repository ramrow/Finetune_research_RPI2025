import torch
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Step 1: Load the dataset
dataset = load_dataset("AdiOO7/llama-2-finance")

# Step 2: Load the tokenizer and model
base_model = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    processing_class=tokenizer,
)
trainer.evaluate()