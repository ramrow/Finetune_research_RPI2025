import pandas as pd

f = pd.read_csv('meshed.csv')    
a = f.dropna()
a.to_csv('meshed.csv', index=False)