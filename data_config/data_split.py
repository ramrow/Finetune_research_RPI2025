from datasets import load_dataset
import pandas as pd
import json
import csv

df = pd.read_json('dataset with blockmesh.json')
shuffled_df = df.sample(frac=1)
shuffled_df.to_json('shuffled_foam.json', index=False, orient='records', indent=1)

percent = 0.2
pivot = int(len(shuffled_df) * percent)
test, train = (shuffled_df[:pivot], shuffled_df[pivot:])
test.to_json("test.json", index=False, orient='records', indent=1)
train.to_json("train.json", index=False, orient='records', indent=1)

# df = pd.read_csv('shuffled_foam.csv')
# tt = pd.read_csv('data/train.csv')
# ts = pd.read_csv('data/test.csv')
# print(len(df), len(tt)+len(ts))