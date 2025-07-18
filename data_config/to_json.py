import pandas as pd

# Read CSV file
df = pd.read_csv('train.csv')

# DataFrame to JSON
df.to_json('train.json', orient='records', lines=True, indent=2)