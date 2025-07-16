import pandas as pd

# Read CSV file
df = pd.read_csv('trd.csv')

# DataFrame to JSON
df.to_json('train.json', orient='records', lines=True)