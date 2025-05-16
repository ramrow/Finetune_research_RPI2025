import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    pipeline,
    logging,
)
from peft import LoraConfig,PeftModel,PeftType
from trl import SFTTrainer, SFTConfig
import types

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            if isinstance(model, PeftModel) and model.peft_type == PeftType.PROMPT_TUNING:
                num_virtual_tokens = model.peft_config["default"].num_virtual_tokens
                shift_logits = outputs.logits[..., :-(1+num_virtual_tokens), :].contiguous()
            else:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
            
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss






base_model = "NousResearch/Llama-2-7b-chat-hf"
# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
# Fine-tuned model
new_model = "llama-7b-python-guanaco"

dataset = load_dataset(guanaco_dataset, split="train")
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


training_args = SFTConfig(
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
    report_to="tensorboard",
    label_names=["text"],
    packing=False,
    max_seq_length=None,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    # dataset_text_field="text",
    # max_seq_length=None,
    compute_loss_func=compute_loss,
    processing_class=tokenizer,
    args=training_args,
    # packing=False,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)