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
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, PeftType


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Compute training loss and additionally compute token accuracies
    """
    (loss, outputs) = super().compute_loss(
        model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
    )

    # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
    if "labels" in inputs and not self.args.use_liger:
        if isinstance(model, PeftModel) and model.peft_type == PeftType.PROMPT_TUNING:
            num_virtual_tokens = model.peft_config["default"].num_virtual_tokens
            shift_logits = outputs.logits[..., :-(1+num_virtual_tokens), :].contiguous()
        else:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
        
        shift_labels = inputs["labels"][..., 1:].contiguous()


base_model = "codellama/CodeLlama-7b-Python-hf"

test_dataset = "mlabonne/guanaco-llama2-1k"

# new_model = "llama-7b-Python-hf-finance-sentiment-recognition"
new_model = "llama-7b-Python-hf-test-guanaco"


dataset = load_dataset(test_dataset, split="train")

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

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    processing_class=tokenizer,
    compute_loss_func=compute_loss,
    # dataset_text_field="text",
    # max_seq_length=None,
    # tokenizer=tokenizer,
    args=training_params,
    # packing=False,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)