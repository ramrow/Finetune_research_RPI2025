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
from transformers import pipeline

model_name_or_path = "llama-7b-python-finance" #path/to/your/model/or/name/on/hub
pipe = pipeline("text-generation", model=model_name_or_path)
print(pipe("### Instruction: What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive} ### Human: Bananas was founded in 1930s and have continued to exist today gaining a revenue of 130%")[0]["generated_text"])

