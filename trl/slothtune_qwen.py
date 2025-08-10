from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


class unsloth_qwen():
    def __init__(self):
        self.md, self.tk =  FastLanguageModel.from_pretrained(
                                model_name= "unsloth/Qwen3-Coder-30B-A3B-Instruct",
                                max_seq_length= 1028,
                                load_in_4bit=True,
                            )
        self.md =  FastLanguageModel.get_peft_model(
                                model= self.md,
                                r= 32,
                                target_modules= "all-linear",
                                bias= "none",
                                lora_dropout= 0.1,
                                lora_alpha=32, #could be 16
                            )
        self.tk.return_tensors = "pt"
        self.tk.pad_token = self.tk.eos_token
        self.tk.padding_side = "right"
        self.ds = (load_dataset("LeoYML/FoamGPT",)).shuffle()
        self.trd = None
        self.tsd = None

    def _helper_(self):
        self.format_data()
        self.finetune()

    def finetune(self):
        training_args = SFTConfig(
            output_dir="./foamqwen",
            num_train_epochs=6,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit", 
            save_steps=250,
            logging_steps=25,
            learning_rate=3e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
            packing=False,
        )
        trainer = SFTTrainer(
            model=self.md,
            train_dataset=self.trd,
            eval_dataset=self.tsd,
            args=training_args,
            processing_class=self.tk,
        )
        # trainer.train()
        trainer.train()
        trainer.model.save_pretrained("foamqwen")
        trainer.processing_class.save_pretrained("foamqwen")
        trainer.evaluate()

    def format_data(self):
        self.trd = (self.ds['train'].map(self.format_data_helper)).remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", 
                                                                                "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])

        self.tsd = (self.ds['test'].map(self.format_data_helper)).remove_columns(["text", "system_prompt", "user_prompt", "folder_name", "file_name", 
                                                                                "case_name", "case_domain", "user_requirement", "file_content", "case_category", "case_solver"])

        # print(self.trd)
        # print(self.tsd)

    def format_data_helper(self, example):
        messages = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example['usr_prompt']},
            {"role": "assistant", "content": example["code_content"]}
        ]
        prompt = self.tk.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tk(prompt, padding="max_length", max_length=1028, truncation=True)
        tokens['labels'] = [
            -100 if token == self.tk.pad_token_id else token for token in tokens['input_ids']
        ]
        return tokens
    

if __name__ == "__main__":
    uq = unsloth_qwen()
    uq._helper_()