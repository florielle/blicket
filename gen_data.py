import pandas as pd

filename = 'data/pokemon.tsv'
train_ratio = 0.7

raw_data = pd.read_csv(filename, delimiter="\t", header=None)
    
filename = filename.strip("data/").strip(".tsv")

train = raw_data.sample(n=int(train_ratio*len(raw_data)),random_state=1111)
val = raw_data[~raw_data.isin(train)].dropna()
train.to_csv(f'data/{filename}_train.tsv', sep="\t", index=False,header=False)
val.to_csv(f'data/{filename}_val.tsv', sep="\t", index=False,header=False)
