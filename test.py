from datasets import load_dataset
import json

def format_data(example):
    prompt = f"Question: {example['description']}"
    response = f"foamfiles:\n{example['foamfiles']}\n\nallrun:\n{example['allrun']}"
    example["text"] = f"{prompt}\n\nResponse:\n{response}"
    return example

ds = load_dataset("YYgroup/NL2FOAM")
data = ds.map(format_data)
d = dict()
d['text'] = data['train']['text']
json_object = json.dumps(d, indent=4)

# Writing to sample.json
with open("processed_foam.json", "w") as outfile:
    outfile.write(json_object)

