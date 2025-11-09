import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,

)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.set_grad_enabled(True)
local_rank = os.getenv("LOCAL_RANK")
device_string = "cuda:" + str(local_rank)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

ds = (load_dataset("LeoYML/FoamGPT",)).shuffle()
model="Qwen/Qwen2.5-7B-Instruct"
new_model = "foamqwen"

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    # device_map="auto",
    device_map={'':device_string},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"

def apply_chat_template(example):
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user",   "content": example["user_prompt"]},
        {"role": "assistant", "content": example["file_content"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,   # we want the answer included
    )
    return {"text": text + tokenizer.eos_token}   # make sure EOS is there

ds = ds.map(
    apply_chat_template,
    remove_columns=ds["train"].column_names,
    num_proc=8,
    desc="Applying chat template",
)

# 2. Create the special collator that masks everything before the assistant
response_template = "<|im_start|>assistant\\n"   # exact string Qwen uses for assistant

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template=response_template,   # loss only after this
    instruction_template=None,            # optional, if you want to mask system+user too
)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64, #change rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",

)

training_args = SFTConfig(
    output_dir="foamqwen",
    num_train_epochs=7,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, 
    optim="paged_adamw_32bit",
    logging_steps=25,
    learning_rate=5.11e-4, #3e-4
    weight_decay=0.03, #0.03
    fp16=False, #False
    bf16=True, #True
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    # dataset_text_field="messages"
)

peft_md = get_peft_model(md, peft_params)

trainer = SFTTrainer(
    model=peft_md,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    args=training_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
trainer.evaluate()