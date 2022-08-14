import pandas as pd
df = pd.read_csv("vgsales.csv")
#print(df.head)
#print(df.sample())
#print(df.tail())
#print(df.info)
#print(df.describe())
#print(df.head())

# CONSULTA COM FILTRO DE DADOS
#print(df.query('Global_Sales>30'))

# CONSULTA ESPECIFICA
#print(df.loc[5,'Global_Sales'])
#print(df.loc [2],[4], [2],[3])

# MOSTRA VALORES DISTINTOS DAS VARIAVEIS - quantidade de dados pela classe
#print(df.nunique())
#print(df['Genre'].unique())

# MOSTRA VALORES Null
#print(df.isnull)
print(df.isnull().sum())

# MOSTRA AS LINHAS QUE ESTÃO MISSING
df2 = df[df['Year'].isnull()]
print (df2)


