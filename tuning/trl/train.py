import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

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
raw_ds = load_dataset("finalform/foamGPT").shuffle()

# NEW: Convert to conversational format (add "messages" column)
def to_conversational(example):
    return {
        "messages": [
            {"role": "system",   "content": example["system_prompt"]},
            {"role": "user",     "content": example["user_prompt"]},
            {"role": "assistant","content": example["file_content"]},
        ]
    }

train_ds = raw_ds["train"].map(to_conversational, remove_columns=raw_ds["train"].column_names)
test_ds  = raw_ds["test"].map (to_conversational, remove_columns=raw_ds["test"].column_names)
# train_ds = ds['train'].map(to_conversational)
# test_ds = ds['test'].map(to_conversational)
# # Optional: Remove unused columns for cleanliness (but keep "messages")
# train_ds = train_ds.remove_columns([
#     "system_prompt", "user_prompt", "folder_name", "file_name", "case_name",
#     "case_domain", "user_requirement", "file_content", "case_category", "case_solver"
# ])
# test_ds = test_ds.remove_columns([
#     "system_prompt", "user_prompt", "folder_name", "file_name", "case_name",
#     "case_domain", "user_requirement", "file_content", "case_category", "case_solver"
# ])

tokenizer.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n\n        {{- '<|im_start|>' + message.role }}\n        {% generation %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- content }}\n            {%- endif %}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"""
print(train_ds)

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

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM
    return_tensors="pt",
)

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
    max_seq_length=2048,               # explicit truncation
    fp16=False,                        # bf16 is already enabled
    remove_unused_columns=False,       # keep "messages"
    label_names=[],                    # <-- **fixes the PeftModel warning**
    ddp_find_unused_parameters=False,  # <-- removes the DDP warning
    dataset_text_field="messages"
    # max_seq_length=1028,  # Add this for truncation
)

# Trainer (use RAW datasets; TRL handles tokenization + masking)
trainer = SFTTrainer(
    model=peft_md,
    train_dataset=train_ds,     # RAW with "messages"
    eval_dataset=test_ds,       # RAW with "messages"
    args=training_args,
    processing_class=tokenizer,        # Handles apply_chat_template internally
    data_collator=data_collator,
    # NO data_collator, formatting_func, or pre-tokenized data!
)

trainer.train()
trainer.model.save_pretrained("foamqwen")
tokenizer.save_pretrained("foamqwen")  # Save tokenizer (not processing_class)
trainer.evaluate()