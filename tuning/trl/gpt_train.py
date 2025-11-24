import torch
from datasets import load_dataset
from transformers import(
    AutoModelForCausalLM, 
    Mxfp4Config,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

# def apply_chat_template(example):
#     messages = [
#         {"role": "system", "content": example["system_prompt"]},
#         {"role": "user", "content": example['user_prompt']},
#         {"role": "assistant", "content": example["file_content"]}
#     ]
#     prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     return {"text": prompt}

# def tokenize_data(example):
#     tokens = tokenizer(example['text'], padding="longest",)
#     # tokens = tokenizer(example['text'], padding="max_length", max_length=1028, truncation=True)
#     tokens['labels'] = [
#         -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
#     ]
#     return tokens
def split_prompt_completion(example):
    return {
        "prompt": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user",   "content": example["user_prompt"]},
        ],
        "completion": [
            {"role": "assistant", "content": example["file_content"]},
        ],
    }


model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
ds = (load_dataset("finalform/foamGPT-old",)).shuffle()
new_model = "foamGPT"

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Apply template
ds = ds.map(
    split_prompt_completion,
    remove_columns=ds["train"].column_names,
    num_proc=8,
)
# train_ds = ds['train'].map(apply_chat_template)
# test_ds = ds['test'].map(apply_chat_template)
# tokenized_train_ds = train_ds.map(tokenize_data)
# tokenized_test_ds = test_ds.map(tokenize_data)
# tokenized_train_ds = tokenized_train_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])
# tokenized_test_ds = tokenized_test_ds.remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_md = get_peft_model(model, peft_config)
# peft_model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="foamGPT",
    # resume_from_checkpoint="./qwen_results/checkpoint-",
    # compute loss every few steps 1.5k/step
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, 
    optim="paged_adamw_32bit",
    # save_steps=750,
    logging_steps=25,
    learning_rate=5.11e-4,
    weight_decay=0.03,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataset_text_field="text",
    assistant_only_loss=True,
    completion_only_loss=True
)

trainer = SFTTrainer(
    model=peft_md,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
trainer.evaluate()