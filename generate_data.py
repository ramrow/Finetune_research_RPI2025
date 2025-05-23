import json
from datasets import load_dataset

dictionary = dict()
dataset = load_dataset("AdiOO7/llama-2-finance", split="train")
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
def process(examples):
    tmp1 = []
    tmp2 = []
    for line in dataset["text"]:
        temp = line.split(" ### Assistant: ")
        tmp1.append(temp[0])
        tmp2.append(label_map[temp[1][:-1]])
        
    dictionary['text'] =  tmp1
    dictionary['label'] = tmp2
    return examples
data = dataset.map(process)
# Serializing json
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
with open("finance_data.json", "w") as outfile:
    outfile.write(json_object)