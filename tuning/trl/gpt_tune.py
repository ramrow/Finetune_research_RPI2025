from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer

def conversational(example):
    return {
        "prompt": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user",   "content": example["user_prompt"]},
        ],
        "completion": [
            {"role": "assistant", "content": example["file_content"]},
        ],
    }

ds = (load_dataset("finalform/foamGPT-old", )).shuffle()
dataset = ds.map(
    conversational,
    remove_columns=ds["train"].column_names,
)

new_model = "foamGPT"
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)

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
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="foamGPT",
    num_train_epochs=5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    learning_rate=5.11e-4,
    gradient_checkpointing=True,
    logging_steps=1,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    report_to="trackio",
    eval_strategy="epoch",
    save_strategy="epoch",
    completion_only_loss= True,
    # push_to_hub=True,
)


trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)
trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
