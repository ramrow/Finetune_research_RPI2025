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
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # Cast to int for safety
device_string = f"cuda:{local_rank}"

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model_name = "Qwen/Qwen2.5-7B-Instruct"
md = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={'': device_string},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
md.config.use_cache = False
md.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and prepare dataset (RAW — NO pre-tokenization!)
ds = load_dataset("finalform/foamGPT").shuffle()

# NEW: Convert to conversational format (add "messages" column)
def to_conversational(example):
    return {
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
            {"role": "assistant", "content": example["file_content"]}  # Include full response
        ]
    }

template =  """ {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {%- generation -%}
            {{- message.content }}
        {%- endgeneration -%}
        {{- '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- generation -%}
            {%- if message.content %}
                {{- '\n' + message.content }}
            {%- endif %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\n<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {{- tool_call.arguments | tojson }}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endgeneration -%}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- generation -%}
    {%- endgeneration -%}
{%- endif %}
"""

train_ds = ds['train'].map(to_conversational)
test_ds = ds['test'].map(to_conversational)
# Optional: Remove unused columns for cleanliness (but keep "messages")
train_ds = train_ds.remove_columns([
    "system_prompt", "user_prompt", "folder_name", "file_name", "case_name",
    "case_domain", "user_requirement", "file_content", "case_category", "case_solver"
])
test_ds = test_ds.remove_columns([
    "system_prompt", "user_prompt", "folder_name", "file_name", "case_name",
    "case_domain", "user_requirement", "file_content", "case_category", "case_solver"
])

tokenizer.chat_template = template

# LoRA config
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
peft_md = get_peft_model(md, peft_params)

# Training args (keep assistant_only_loss=True — now it works!)
training_args = SFTConfig(
    output_dir="foamqwen",
    num_train_epochs=7,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    assistant_only_loss=True,  # Now valid!
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
    packing=False,  # Recommended for clean masking
    eval_strategy="epoch",
    save_strategy="epoch",
    # max_seq_length=1028,  # Add this for truncation
)

# Trainer (use RAW datasets; TRL handles tokenization + masking)
trainer = SFTTrainer(
    model=peft_md,
    train_dataset=train_ds,     # RAW with "messages"
    eval_dataset=test_ds,       # RAW with "messages"
    args=training_args,
    processing_class=tokenizer,        # Handles apply_chat_template internally
    # NO data_collator, formatting_func, or pre-tokenized data!
)

trainer.train()
trainer.model.save_pretrained("foamqwen")
tokenizer.save_pretrained("foamqwen")  # Save tokenizer (not processing_class)
trainer.evaluate()