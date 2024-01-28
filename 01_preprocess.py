import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA = sys.argv[1] #"data/NFLXcsv"
train_output_file = sys.argv[2] #"data/preprocessed/cardio_train.csv"
test_output_file = sys.argv[3] # "data/preprocessed/cardio_test.csv"

df = pd.read_csv(DATA)

df_train = df[:-1000]
df_test = df[-1000:]
df_train = df_train[["Date", "Close"]]
df_test = df_test[["Date", "Close"]]

df_train.to_csv(train_output_file, index=False)
df_test.to_csv(test_output_file, index=False)
