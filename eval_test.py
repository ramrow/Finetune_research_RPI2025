from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from trl import SFTConfig,SFTTrainer
from peft import LoraConfig
import numpy as np
import evaluate
import torch

# import wandb
# wandb.init(mode="disabled")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'].float(), inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss



temp_data = load_dataset("AdiOO7/llama-2-finance", split="train")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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
    model_inputs = tokenizer(dataset["text"], padding="max_length", truncation=True, max_length=64) #max_length=2

    labels = tokenizer(text_target=dataset["class"], padding="max_length", truncation=True, max_length=64,) #max_length=2

    model_inputs["label"] = labels["input_ids"]
    model_inputs["label"] = np.array(model_inputs["label"]).astype(int)

    return model_inputs


def compute_metrics(eval_pred, compute_result):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    print(logits, labels)
    print(np.array(logits).shape, np.array(labels).shape)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = temp_data.map(preprocess_function, batched=True)
print(tokenized_datasets)
model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",num_labels=64)
training_args = TrainingArguments(output_dir="./pre_result", per_device_eval_batch_size=1)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets,
    compute_metrics=compute_metrics,
)
trainer.evaluate()