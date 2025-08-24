import pandas as pd


def drop_nan(name):
    path = "data/" + name
    f = pd.read_csv(path)
    a = f.dropna()
    a.to_csv(name, index=False)

drop_nan("train.csv")
drop_nan("test.csv")