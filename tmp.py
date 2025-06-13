from datasets import load_dataset
import json

names = ['0/nuTilda', 'system/controlDict', 'system/fvSchemes', '0/p', 'constant/turbulenceProperties', 'system/fvSolution', '0/nut', '0/k', '0/U', 'constant/transportProperties', '0/epsilon', '0/sigma', 'constant/fvOptions', '0/omega', '0/s', 'constant/MRFProperties', 'constant/dynamicMeshDict', 'system/topoSetDict'] 

data = json.load(open("processed_foam.json"))
d = dict()
d['text'] = data['text']
d['allrun'] = data['allrun']
for n in names:
    d[n] = []
    for f in data[n]:
        d[n].append(str(f))

json_object = json.dumps(d, indent=4)
with open("processed_foam.json", "w") as outfile:
    outfile.write(json_object)