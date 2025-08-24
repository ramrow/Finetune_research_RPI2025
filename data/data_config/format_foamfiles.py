from datasets import load_dataset
import pandas as pd
import json

names = {'0/nuTilda', 'system/controlDict', 'system/fvSchemes', '0/p', 'constant/turbulenceProperties', 
        'system/fvSolution', '0/nut', '0/k', '0/U', 'constant/transportProperties', '0/epsilon', 
        '0/sigma', 'constant/fvOptions', '0/omega', '0/s', 'constant/MRFProperties', 'constant/dynamicMeshDict', 'system/topoSetDict'} 


def format_data(example):
    prompt = f"<s>[INST] {example['description']}"
    example['text'] = prompt
    return example

ds = load_dataset("YYgroup/NL2FOAM")
data = ds.map(format_data)
d = dict()
d['text'] = data['train']['text']
d['allrun'] = data['train']['allrun']

for n in names:
    d[n] = []

i = 0
length = len(data['train']['file_tree'])
for line in open('foamfiles.json','r'):
    if(i == length):
        break
    tmp = data['train']['file_tree'][i]
    info = json.loads(line)
    for file in tmp:
        d[file].append(info[file])
    left_over = list(names.difference(set(tmp)))
    for file in left_over:
        d[file].append('NONE')
    i += 1

json_object = json.dumps(d, indent=4)
with open("processed_foam.json", "w") as outfile:
    outfile.write(json_object)


''''''
array = []

def format_data(example):
    example['text'] = example['foamfiles']
    tmp = set(example['file_tree']).difference(set(array))
    array.extend(list(tmp))
    return example

ds = load_dataset("YYgroup/NL2FOAM")
data = ds.map(format_data)
print(array)
''''''


''''''
def format_data(example):
    example['text'] = example['foamfiles']
    return example

ds = load_dataset("YYgroup/NL2FOAM")
data = ds.map(format_data)

d = data['train']['text']
json_object = json.dumps(d, indent=4)

file = open("foamfiles.json",'w')
for line in d:
    file.write(line + "\n")
''''''