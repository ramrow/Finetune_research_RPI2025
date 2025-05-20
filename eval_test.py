from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from trl import SFTConfig,SFTTrainer
from peft import LoraConfig
import numpy as np
import evaluate
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'].float(), inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss



temp_data = load_dataset("AdiOO7/llama-2-finance", split="train")

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

metric = evaluate.load("exact_match")

def preprocess_function(examples):
    dataset = dict()
    tmp1 = []
    tmp2 = []
    for line in examples["text"]:
        temp = line.split(" ### Assistant: ")
        tmp1.append(temp[0])
        tmp2.append(temp[1][:-1])
    dataset["text"] = tmp1
    dataset["class"] = tmp2
    model_inputs = tokenizer(dataset["text"], padding="max_length", truncation=True, max_length=512)

    labels = tokenizer(text_target=dataset["class"], padding="max_length", truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(logits)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = temp_data.map(preprocess_function, batched=True)
print(tokenized_datasets)
model = AutoModelForSequenceClassification.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
training_args = TrainingArguments(output_dir="./pre_result", per_device_eval_batch_size=1)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets,
    compute_metrics=compute_metrics,
)
trainer.evaluate()