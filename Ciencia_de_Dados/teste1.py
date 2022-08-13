# importacoes
import pickle
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset0 = pd.read_csv("Heart_data2.csv")
dataset1 = dataset0.replace('',np.nan)

# a = a.dropna(axis="columns", how="any")

#dataset = [x for x in dataset0 if pd.isnull(x) == False]
#dataset = [x for x in dataset0 if pd.isnull(x) == False and x != 'nan']

print (dataset)
# TOP 5
#print(dataset.head())

# finais
#print((dataset.tail()))