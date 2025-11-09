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

def apply_chat_template(example):
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example['user_prompt']},
        {"role": "assistant", "content": example["file_content"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt}

def tokenize_data(example):
    # tokens = tokenizer(example['text'], padding="longest",)
    tokens = tokenizer(example['text'], padding="max_length", max_length=1028, truncation=True)
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens


ds = (load_dataset("finalform/foamGPT",)).shuffle()
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
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n\n        {{- '<|im_start|>' + message.role }}\n        {% generation %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- content }}\n            {%- endif %}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"""

train_ds = ds['train'].map(apply_chat_template)
test_ds = ds['test'].map(apply_chat_template)
tokenized_train_ds = train_ds.map(tokenize_data)
tokenized_test_ds = test_ds.map(tokenize_data)
tokenized_train_ds = tokenized_train_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])
tokenized_test_ds = tokenized_test_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64, #change rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",

)

response_template = "<|im_start|>assistant"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

training_args = SFTConfig(
    output_dir="foamqwen",
    # resume_from_checkpoint="./qwen_results/checkpoint-",
    # compute loss every few steps 1.5k/step
    num_train_epochs=7,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, 
    assistant_only_loss=True,
    # gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    # save_steps=750,    
    logging_steps=25,
    learning_rate=5.11e-4, #3e-4
    weight_decay=0.03, #0.03
    fp16=False, #False
    bf16=True, #True
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
)

peft_md = get_peft_model(md, peft_params)

trainer = SFTTrainer(
    model=peft_md,
    # data_collator=collator,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
trainer.evaluate()