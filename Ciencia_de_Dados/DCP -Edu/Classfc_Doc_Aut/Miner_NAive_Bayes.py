import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#DEFININDO SOMENTE 4 CATEGORIAS
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

#TREINAMENTO
twenty_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
#print(twenty_train)

#CLASSES
print(twenty_train.target_names)

#VISUALIZAR 10 LINHAS DO REGISTRO
print(len(twenty_train.data))
print("\n".join(twenty_train.data[0].split("\n")[:10]))

#VISUALIZAR VARIAVEIS TARGET
print(twenty_train.target_names[twenty_train.target[3]])

#O SCIKIT_LEARN REGISTRA OS LABELS COMO ARRAY DE NUMEROS, AFIM DE AUMENTAR VELOCIDADE
print(twenty_train.target[:10])

#VISUALIZAR AS CLASSES DOS 10 PRIMEIROS REGISTROS
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
# TOKENIZING
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect.vocabulary_.get(u'algorithm')
print(X_train_counts.shape)

# DE OCORRENCIAS A FREQUENCIAS - TERM FREQUENCY TIMES INVERSE DOCUMENT FREQUENCY (Tfidf)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# D EOUTRA FORMA COMBINANDO AS FUNCOES
tfidf_transformer = TfidfTransformer()
X_train_tfidf =tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# CRIANDO O MODELO MULTINOMIAL
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
print(clf)