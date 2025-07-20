import json
import pandas as pd

f=pd.read_csv("data/test.csv")
keep_col = ['usr_prompt', 'code_content', 'system_prompt',]
new_f = f[keep_col]
new_f.to_csv("data/test.csv", index=False, )