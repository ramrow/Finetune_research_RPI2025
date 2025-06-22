import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

os.environ["CUDA_VISIBLE_DEVICES"]="0"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=True,
)
def format_files(example):
    return {
        "0/nuTilda": example["0/nuTilda"],
        "system/controlDict": example["system/controlDict"],
        "system/fvSchemes": example["system/fvSchemes"],
        "0/p": example["0/p"],
        "constant/turbulenceProperties": example["constant/turbulenceProperties"],
        "system/fvSolution": example["system/fvSolution"],

        "0/nut": example["0/nut"],
        "0/k": example["0/k"],
        "0/U": example["0/U"],
        "constant/transportProperties": example["constant/transportProperties"],
        "0/epsilon": example["0/epsilon"],
        "0/sigma": example["0/sigma"],
        "constant/fvOptions": example["constant/fvOptions"],
        "0/omega": example["0/omega"],
        "0/s": example["0/s"],
        "constant/MRFProperties": example["constant/MRFProperties"],
        "constant/dynamicMeshDict": example["constant/dynamicMeshDict"],
        "system/topoSetDict": example["system/topoSetDict"],
        "allrun": example["allrun"]
    }


def apply_chat_template(example):
    messages = [
        {"role": "user", "content": example['text']},
        {"role": "assistant", "content": format_files(example)}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt}

def tokenize_data(example):
    tokens = tokenizer(example['text'], padding="longest", return_tensors="pt")
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

ds = (load_dataset("finalform/processed_foam", split="train")).shuffle()
model="codellama/CodeLlama-7b-Instruct-hf"
new_model = "llama-foam"

md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    # device_map={"": 0}
    device_map="auto"
)
md.config.use_cache = False
md.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.return_tensors = "pt"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

organized_ds = ds.map(apply_chat_template)
tokenized_ds = organized_ds.map(tokenize_data)
tokenized_ds = tokenized_ds.remove_columns(['text', 'allrun', '0/U', 'constant/transportProperties', 'constant/turbulenceProperties', '0/s', '0/sigma', 'constant/fvOptions', '0/omega', 'constant/MRFProperties', '0/k', 'system/fvSchemes', '0/nut', '0/p', '0/epsilon', 'system/controlDict', 'system/fvSolution', 'constant/dynamicMeshDict', '0/nuTilda', 'system/topoSetDict'])
tokenized_ds = tokenized_ds.train_test_split(0.05)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./llama_results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    # per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
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
    report_to="tensorboard",
    packing=False,
)

peft_md = get_peft_model(md, peft_params)
# peft_md = dispatch_model(peft_md, device_map={})

trainer = SFTTrainer(
    model=peft_md,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    # peft_config=peft_params,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)
trainer.evaluate()


        # {"role": "0/nuTilda", "content": example['0/nuTilda']},
        # {"role": "system/controlDict", "content": example['system/controlDict']},
        # {"role": "system/fvSchemes", "content": example['system/fvSchemes']},
        # {"role": "0/p", "content": example['0/p']},
        # {"role": "constant/turbulenceProperties", "content": example['constant/turbulenceProperties']},
        # {"role": "system/fvSolution", "content": example['system/fvSolution']},
        # {"role": "0/nut", "content": example['0/nut']},
        # {"role": "0/k", "content": example['0/k']},
        # {"role": "0/U", "content": example['0/U']},
        # {"role": "constant/transportProperties", "content": example['constant/transportProperties']},
        # {"role": "0/epsilon", "content": example['0/epsilon']},
        # {"role": "0/sigma", "content": example['0/sigma']},
        # {"role": "constant/fvOptions", "content": example['constant/fvOptions']},
        # {"role": "0/omega", "content": example['0/omega']},
        # {"role": "0/s", "content": example['0/s']},
        # {"role": "constant/MRFProperties", "content": example['constant/MRFProperties']},
        # {"role": "constant/dynamicMeshDict", "content": example['constant/dynamicMeshDict']},
        # {"role": "system/topoSetDict", "content": example['system/topoSetDict']},
        # {"role": "allrun", "content": example['allrun']}