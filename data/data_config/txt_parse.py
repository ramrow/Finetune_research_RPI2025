import json
import pandas as pd




_0file = ['alphat','epsilon', 'k','nut','p','p_rgh','T','U']
_constantfile = ['g', 'momentumTransport', 'physicalProperties']
_systemfile = ['blockMeshDict', 'controlDict', 'fvSchemes', 'fvSolution']

d = dict()
d['user_prompt'] = []
d['system_prompt'] = []
d['file_content'] = []

for name in _0file:
    link1 = "test/0/" + name  + '.txt'
    link2 = "C:\\Users\\Peijing Xu\\projects\\yue_research\\main_finetune\\test\\0\\" + name
    with open(link1, 'r') as file:
        file_content = file.read()
    prompts = file_content.split("code user prompt:\n")
    temp = prompts[0].replace('code system prompt:\n','')

    d['user_prompt'].append(prompts[1])
    d['system_prompt'].append(temp)
    with open(link2, 'r') as file:
        file_content = file.read()
    d['file_content'].append(file_content)

for name in _constantfile:
    link1 = "test/constant/" + name  + '.txt'
    link2 = "C:\\Users\\Peijing Xu\\projects\\yue_research\\main_finetune\\test\\constant\\" + name
    with open(link1, 'r') as file:
        file_content = file.read()
    prompts = file_content.split("code user prompt:\n")
    temp = prompts[0].replace('code system prompt:\n','')

    d['user_prompt'].append(prompts[1])
    d['system_prompt'].append(temp)
    with open(link2, 'r') as file:
        file_content = file.read()
    d['file_content'].append(file_content)

for name in _systemfile:    
    link1 = "test/system/" + name  + '.txt'
    link2 = "C:\\Users\\Peijing Xu\\projects\\yue_research\\main_finetune\\test\\system\\" + name
    with open(link1, 'r') as file:
        file_content = file.read()
    prompts = file_content.split("code user prompt:\n")
    temp = prompts[0].replace('code system prompt:\n','')    

    d['user_prompt'].append(prompts[1])
    d['system_prompt'].append(temp)
    with open(link2, 'r') as file:
        file_content = file.read()
    d['file_content'].append(file_content)

print(len(d['user_prompt']), len(d['system_prompt']), len(d['file_content']))
df = pd.DataFrame(d)
for index, row in df.iterrows():
    tmp = row.system_prompt
    new_prompt = tmp.replace("codeâ€”no", "code-no")
    row.system_prompt = new_prompt

df.to_csv('bernard.csv', index=False)
