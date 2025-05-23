from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig

import numpy as np
import evaluate
import torch

# import wandb
# wandb.init(mode="disabled")

data = load_dataset("/kaggle/input/finance-data")
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'].float(), inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss


tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
metric = evaluate.load("accuracy")
def tokenization(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_dataset = data.map(tokenization, batched=True)
print(tokenized_dataset)
print(np.array(tokenized_dataset['train']['attention_mask']).shape)


def compute_metrics(eval_pred, compute_result):
    logits, labels = eval_pred
    print(logits)
    predictions = np.argmax(logits, axis=1)
    print(predictions, labels)
    print(np.array(logits).shape, np.array(labels).shape)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",num_labels=64)
training_args = TrainingArguments(output_dir="./pre_result", per_device_eval_batch_size=1)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)
trainer.evaluate()
# def preprocess_function(examples):
#     dataset = dict()
#     tmp1 = []
#     tmp2 = []
#     for line in examples["text"]:
#         temp = line.split(" ### Assistant: ")
#         tmp1.append(temp[0])
#         tmp2.append(temp[1][:-1])
#     dataset["text"] = tmp1
#     dataset["class"] = tmp2
#     model_inputs = tokenizer(dataset["text"], padding="max_length", truncation=True, max_length=64) #max_length=2

#     labels = tokenizer(text_target=dataset["class"], padding="max_length", truncation=True, max_length=64) #max_length=2

#     model_inputs["label"] = labels["input_ids"]
#     model_inputs["label"] = np.array(model_inputs["label"]).astype(int)

#     return model_inputs


# def compute_metrics(eval_pred, compute_result):
#     logits, labels = eval_pred
#     print(logits)
#     predictions = np.argmax(logits, axis=1)
#     print(predictions, labels)
#     print(np.array(logits).shape, np.array(labels).shape)
#     return metric.compute(predictions=predictions, references=labels)

# # tokenized_datasets = temp_data.map(preprocess_function, batched=True)
# model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",num_labels=64)
# training_args = TrainingArguments(output_dir="./pre_result", per_device_eval_batch_size=1)

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     eval_dataset=dataset,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
# )
# trainer.evaluate()




























# from transformers import pipeline
# from datasets import load_dataset
# from evaluate import evaluator
# import evaluate

# pipe = pipeline("text-classification", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=0)
# temp_data = load_dataset("AdiOO7/llama-2-finance", split="train")
# def process(examples):
#     tmp1 = []
#     tmp2 = []
#     for line in temp_data["text"]:
#         temp = line.split(" ### Assistant: ")
#         tmp1.append(temp[0])
#         tmp2.append(temp[1][:-1])
        
#     examples['text'] =  tmp1
#     examples['label'] = tmp2
#     return examples
# data = temp_data.map(process)
# metric = evaluate.load("accuracy")
# task_evaluator = evaluator("text-classification")
# results = task_evaluator.compute(model_or_pipeline=pipe, data=data, metric=metric, label_mapping={"negative": 0, "neutral": 1, "positive": 2},)
# print(results)
















# import evaluate

# from evaluate import evaluator

# from transformers import pipeline

# from datasets import load_dataset

# # Load a pre-trained text classification pipeline
# # Using a smaller model for potentially faster execution

# try:

#    pipe = pipeline("text-classification", model="AdiOO7/llama-2-finance", device=0) # Use CPU

# except Exception as e:

#    print(f"Could not load pipeline: {e}")

#    pipe = None

# if pipe:

#    # Load a small subset of the IMDB dataset

#    try:

#        data = load_dataset("AdiOO7/llama-2-finance", split="train") # Smaller subset for speed

#    except Exception as e:

#        print(f"Could not load dataset: {e}")

#        data = None

#    if data:

#        # Load the accuracy metric

#        accuracy_metric = evaluate.load("accuracy")

#        # Create an evaluator for the task

#        task_evaluator = evaluator("text-classification")

#        # Correct label_mapping for IMDB dataset

#        label_mapping = {

#            'negative': -1,  # Map NEGATIVE to -1

#            'neutral': 0,

#            'positive': 1   # Map POSITIVE to 1

#        }

#        # Compute results

#        eval_results = task_evaluator.compute(

#            model_or_pipeline=pipe,

#            data=data,

#            metric=accuracy_metric,

#            input_column="text",  # Specify the text column

#            label_column="label", # Specify the label column

#            label_mapping=label_mapping  # Pass the corrected label mapping

#        )

#        print("\nEvaluator Results:")

#        print(eval_results)

#        # Compute with bootstrapping for confidence intervals

#        bootstrap_results = task_evaluator.compute(

#            model_or_pipeline=pipe,

#            data=data,

#            metric=accuracy_metric,

#            input_column="text",

#            label_column="label",

#            label_mapping=label_mapping,  # Pass the corrected label mapping

#            strategy="bootstrap",

#            n_resamples=10  # Use fewer resamples for faster demo

#        )

#        print("\nEvaluator Results with Bootstrapping:")

#        print(bootstrap_results)