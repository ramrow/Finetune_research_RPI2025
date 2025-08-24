import pandas as pd
import csv
import json

# Read CSV file
# df = pd.read_csv('data/train.csv')

# # DataFrame to JSON
# df.to_json('data/train.json', orient='records', lines=True)

with open('data/test.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    data = list(csv.DictReader(csvfile))

with open('data/test.json', mode='w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=4)