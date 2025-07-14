from datasets import load_dataset
import pandas as pd
import json
import csv

df = pd.read_csv('formatted_dataset.csv')
shuffled_df = df.sample(frac=1)
shuffled_df.to_csv('shuffled_foam.csv', index=False)

percent = 0.3
pivot = int(len(shuffled_df) * percent)
test, train = (shuffled_df[:pivot], shuffled_df[pivot:])
test.to_csv("test.csv", index=False)
train.to_csv("train.csv", index=False)

# df = pd.read_csv('shuffled_foam.csv')
# tt = pd.read_csv('data/train.csv')
# ts = pd.read_csv('data/test.csv')
# print(len(df), len(tt)+len(ts))