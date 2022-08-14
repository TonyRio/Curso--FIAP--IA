import pandas as pd
df = pd.read_csv("vgsales.csv")
#print(df.head)
#print(df.sample())
#print(df.tail())
#print(df.info)
#print(df.describe())
print(df.head())

# CONSULTA COM FILTRO DE DADOS
print(df.query('Global_Sales>30'))

# CONSULTA ESPECIFICA
#print(df.loc[5,'Global_Sales'])


