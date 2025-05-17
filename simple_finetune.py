from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("hetalshah1981/llama2_finetune_offerings")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5, 
    per_device_train_batch_size=16,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()
trainer.model.save_pretrained("llama-7b-python-simple")
trainer.tokenizer.save_pretrained("llama-7b-python-simple")
results = trainer.evaluate()
print(f"Validation Accuracy: {results['eval_accuracy']}")