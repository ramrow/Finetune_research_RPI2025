import pandas as pd

f = pd.read_json('dataset_v2.json')
k = f.loc[f['file_name'] == 'blockMeshDict']
k.to_csv('meshed.csv', index=False,)