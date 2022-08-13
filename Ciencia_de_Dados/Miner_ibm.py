# importacoes
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("Heart_data2.csv")
# TOP 5
print(dataset.head())

# finais
print((dataset.tail()))

# shape
print(dataset.shape)

# chacagem de valores nulos (NaN)
print(dataset.isnull().sum())

# tipos dos dados
print(dataset.dtypes)

# descricao
print(dataset.describe().T)

# target (ALvo desejado)
print("Pessoas com doencas cardiacas: ", dataset['TenYearCHD'].value_counts()[1])
print("Pessoas sem doencas cardiacas: ", dataset['TenYearCHD'].value_counts()[0])
print("\n")

# Quantos valores unicos tem em cada variavel
for feature in dataset.columns:
    print(feature, " : ", len(dataset[feature].unique()))
# Mapa de correlacao

print(plt.figure(figsize=(12,8)))
print(sns.heatmap(dataset.corr(), annot=True, cmap = "YlGnBu"))
print(plt.show())

# dependencia e independencia das Features

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print(X.head())
print(y.head())

# TREINAMENTO TEST SPLIT
#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#print(x_train.shape)
#print(x_test.shape)

# Importando metricas de performance
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# CRIANDO AS MAQUINAS PREDITIVAS
# RandomForecastClassifier:
from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier()
#RandomForest = RandomForest.fit(x_train, y_train)

# PREDICAO

#y_pred = RandomForest.predict(x_test)

# Performance

#print("Acuracia : ", accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# criando o pickle file
filename = "Heart.pkl"
pickle.dump(RandomForest, open(filename, "wb"))
