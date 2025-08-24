from datasets import load_dataset
import json

texts = []
labels = []


def format_data(example):
    prompt = f"{example['description']}"
    response = f"{example['foamfiles']}\n\n{example['allrun']}"

    example['text'] = prompt
    example['labels'] = response
    # example["text"] = f"<s>[INST] {prompt}\n\n[/INST]{response}"

    texts.append(prompt)
    labels.append(response)

    return example

ds = load_dataset("YYgroup/NL2FOAM")
data = ds.map(format_data)
d = dict()
d['text'] = texts
d['label'] = labels

print(len(d['text']), len(d['label']))
json_object = json.dumps(d, indent=4)

with open("processed_foam.json", "w") as outfile:
    outfile.write(json_object)