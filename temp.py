from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
import numpy as np
import evaluate
import torch



import numpy as np

def function1(examples):
    tmp1 = []
    tmp2 = []
    for line in examples["text"]:
        temp = line.split(" ### Assistant: ")
        print(temp)
        tmp1.append(temp[0])
        tmp2.append(temp[1][:-1])

    examples['text'] =  tmp1
    examples['label'] = tmp2
    return examples
# ls = [[ 0 , 1 , 2 , 3], [ 4 , 5 , 6 , 7], [ 8 , 9 , 10 , 11]]
# print(np.argmax(ls, axis=2))
dataset = load_dataset("AdiOO7/llama-2-finance", split="train")
print(dataset.map(function1))
# tmp = []
# for line in dataset["text"]:
#     temp = line.split(" ### Assistant: ")
#     tmp.append(temp[0])
#     # tmp.append(temp[1][:-1])

# print(tmp)