#importacoes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("Heart_data2.csv")
# TOP 5
print(dataset.head())

# finais
print((dataset.tail()))

#shape
print(dataset.shape)

#chacagem de valores nulos (NaN)
print(dataset.isnull().sum())

#tipos dos dados
print(dataset.dtypes)

#descricao
print(dataset.describe().T)

#target (ALvo desejado)
print("Pessoas com doencas cardiacas: ", dataset['TenYearCHD'].value_counts()[1])
print("Pessoas sem doencas cardiacas: ", dataset['TenYearCHD'].value_counts()[0])
print("\n")

#Quantos valores unicos tem em cada variavel
for feature in dataset.columns:
    print(feature, " : ", len(dataset[feature].unique()))
# Mapa de correlacao

print(plt.figure(figsize=(12,8)))
print(sns.heatmap(dataset.corr(), annot=True, cmap = "YlGnBu"))
print(plt.show())
