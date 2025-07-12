import os
import torch
import logging
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_scheduler,
    )
from torch.utils.data import( 
    DataLoader,
    )
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

accelerator = Accelerator()
# def record(metrics, locals, accelerator, save_file="metrics.json"):
#     accelerator.wait_for_everyone()
#     local_stats = torch.tensor([locals["loss_sum"], locals["corrects_sum"], locals["valid_toks"], locals["train_step"]], device=accelerator.device)
#     # global_loss, global_corrects_sum, global_valid_toks, global_train_step = accelerator.reduce(local_stats, reduction="sum")
#     # if accelerator.is_main_process:
#     #     avg_loss = global_loss.item() / global_train_step.item()
#     #     metrics["loss"].append(avg_loss)
#     #     metrics["accuracy"].append(global_corrects_sum.item() / global_valid_toks.item())
#     #     metrics["steps"].append(global_train_step.item())
#     #     logging.info(f"Current step's ({locals["train_step"]}) average loss is {avg_loss:.4f}")
#     #     dicts.save_as_json(metrics, save_file)
#     accelerator.wait_for_everyone()
#     return metrics

class torch_prep():
    class CustomSFTTrainer(SFTTrainer):
        def custom_train(self, optimizer, lr_sch, ds, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
            NUM_EPOCHES=self.args.num_train_epochs
            num_training_steps = len(ds) * NUM_EPOCHES
            print(f"total training steps: {num_training_steps}")
            self.optimizer=optimizer
            self.lr_scheduler=lr_sch
            # model = self.model
            for epoch in range(NUM_EPOCHES):
                self.model.train()
                logging.info(f"Epoch {epoch + 1} of {NUM_EPOCHES}")
                process_idx = accelerator.process_index
                # metrics = {"loss": [], "accuracy": [], "steps": []}
                locals = {"loss_sum": 0.0, "corrects_sum": 0, "valid_toks": 0, "train_step": 0}
                for step, batch in enumerate(ds):   
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    predictions = outputs.logits.argmax(dim=-1)
                    valids_mask = batch["labels"] != -100
                    corrects = (predictions[valids_mask] == batch["labels"][valids_mask]).sum().item()
                    locals["corrects_sum"] += corrects
                    locals["valid_toks"] += valids_mask.sum().item()
                    locals["loss_sum"] += loss.item()
                    locals["train_step"] = step + 1

                    if step % self.args.logging_steps == 0:
                        logging.info(f"{process_idx}: train step number {step}")
                        print(f"Step {step}: Loss = {loss.item()}")
                        print(locals)

            print("Training is Done")

    def __init__(self):
        self.model_name = "Qwen/Qwen-7B"
        self.new_model = "qwen-foam"
        self.data_name = "finalform/split_foam"

        self.output_dir="./qwen_results",
        self.num_train_epochs=1,
        self.per_device_train_batch_size=3,
        self.per_device_eval_batch_size=3,
        self.gradient_accumulation_steps=8,
        self.optim="paged_adamw_32bit",
        self.save_steps=250,
        self.logging_steps=25,
        self.learning_rate=3e-4,
        self.weight_decay=0.01,
        self.fp16=False,
        self.bf16=True,
        self.max_grad_norm=0.3,
        self.max_steps=-1,
        self.warmup_ratio=0.03,
        self.group_by_length=True,
        self.lr_scheduler_type="constant",
        self.report_to="tensorboard",
        self.packing=False,


    def pre_loading(self):

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=32, #change rank
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
        md = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        md.config.use_cache = False
        md.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.return_tensors = "pt"
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.eos_token = '<|endoftext|>'
        tokenizer.padding_side = "right"

        tokenizer.chat_template =   "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}" \
                                    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ " \
                                    "'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% if " \
                                    "loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
        ds = (load_dataset("finalform/split_foam",))
        peft_md =  get_peft_model(md, peft_params)
        return peft_md, tokenizer, ds['train'], ds['test']
    
    def prep_(self, md, tk, train, test):

        def apply_chat_template(example):
            messages = [
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": example['usr_prompt']},
                {"role": "assistant", "content": example["code_content"]}
            ]
            prompt = tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"text": prompt}
        def tokenize_data(example):
            tokens = tk(example['text'], padding="max_length", max_length=1028, truncation=True)
            tokens['labels'] = [
                -100 if token == tk.pad_token_id else token for token in tokens['input_ids']
            ]
            return tokens

        data_collator = DataCollatorForLanguageModeling(tokenizer=tk, mlm=False)
        optimizer = torch.optim.AdamW(md.parameters(), lr=self.learning_rate[0], weight_decay=self.weight_decay[0])
        lr_scheduler = get_scheduler("constant", optimizer=optimizer)

        train_ds = ((train.map(apply_chat_template)).map(tokenize_data)).remove_columns(["text", "system_prompt", "usr_prompt", "folder_name", "file_name", "case_path", "description", "code_content"])
        test_ds = ((test.map(apply_chat_template)).map(tokenize_data)).remove_columns(["text", "system_prompt", "usr_prompt", "folder_name", "file_name", "case_path", "description", "code_content"])
        train_dl = DataLoader(train_ds, self.per_device_train_batch_size, shuffle=False, collate_fn=data_collator)
        test_dl = DataLoader(test_ds, self.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)

        train_dl = accelerator.prepare(train_dl)
        test_dl = accelerator.prepare(test_dl)

        model, optimizer, lr_scheduler = accelerator.prepare(
            md, optimizer, lr_scheduler
        )

        return model, tk, optimizer, lr_scheduler, train_dl, test_dl, train_ds, test_ds
    
    def train_(self, md, tk, optimizer, lr_sch, train_dl, test_dl, train_ds, test_ds):

        training_args = SFTConfig(
                                output_dir="./qwen_results",
                                # resume_from_checkpoint="./qwen_results/checkpoint-",
                                # compute loss every few steps 1.5k/step
                                num_train_epochs=self.num_train_epochs,
                                per_device_train_batch_size=self.per_device_train_batch_size,
                                per_device_eval_batch_size=self.per_device_eval_batch_size,
                                gradient_accumulation_steps=self.gradient_accumulation_steps,
                                optim=self.optim,
                                save_steps=self.save_steps,
                                logging_steps=self.logging_steps,
                                learning_rate=self.learning_rate,
                                weight_decay=self.weight_decay,
                                fp16=self.fp16,
                                bf16=self.bf16,
                                max_grad_norm=self.max_grad_norm,
                                max_steps=self.max_steps,
                                warmup_ratio=self.warmup_ratio,
                                group_by_length=self.group_by_length,
                                lr_scheduler_type=self.lr_scheduler_type,
                                report_to=self.report_to,
                                packing=self.packing,
        )

        trainer = self.CustomSFTTrainer(
            model=md,
            processing_class=tk,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            args=training_args,
        )
        trainer.custom_train(optimizer, lr_sch, train_dl)
        trainer.model.save_pretrained(self.new_model)
        trainer.processing_class.save_pretrained(self.new_model)
        trainer.evaluate()

    def process_(self):
        md, tk, train, test = self.pre_loading()
        md_, tk_, opt, sch, trl, tsl, trd, tsd = self.prep_(md, tk, train, test)
        self.train_(md_, tk_, opt, sch, trl, tsl, trd, tsd)

if __name__ == "__main__":
    tt = torch_prep()
    tt.process_()
