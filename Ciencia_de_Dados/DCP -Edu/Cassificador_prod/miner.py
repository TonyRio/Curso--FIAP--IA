import gzip
import json
import warnings
import gensim
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk

#Carregar Data Set

corpus = list()
with gzip.open('ecommerce.json.gz') as fp:
    for line in fp:
        entry =line.decode('utf8')
        corpus.append(json.loads(entry))
#print(corpus[0]['name'])
print(len(corpus))
#print(gensim.summarization.summarize(corpus[2]['descr']))

#construindo um datase com somente 10.000 primeiros produtos
dataset = list()
for entry in corpus[:10000]:
    if 'cat' in entry:
        dataset.append(((entry['name'], entry['cat'.lower().strip()])))
#print (dataset)
#print(len(dataset))
#print(dataset[:2])

# QUANTAS CATEGORIAS DISTINTAS NOS TEMOS E QUANTOS ITENS POR CATEGORIA
counter = Counter([cat for prod, cat in dataset])
print(counter.most_common())

stopwords = nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)

#CONSTRUINDO MODELOS PREDITIVOS

# MODELO SVM COM PIPELINE
modelo = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel = 'linear', probability = True))])
LabelEncoder

# OBJETO PARA NORMALIZACAO DE LABELS
encoder = LabelEncoder()
encoder

# OBTENDO DADOS DAS LABELS
data = [prod for prod, cat in dataset]
labels = [cat for prod, cat in dataset]
print (len(data))

# NORMALIZACAO DO LABELS
target = encoder.fit_transform(labels)
print (target)
print(encoder.classes_.item(2))

# FIT DO MODELO
modelo.fit(data, target)

# PREVENDO CATEGORIA A PARTIR DA DESCRICAO
print (modelo.predict(["caneta"]))
print(encoder.classes_[11])

# PROBABILIDADE DO PRODUTO
probs =modelo.predict_proba(['INTEL'])
print(probs)

# PROBABILIDADE DE CATEGORIAS PARA O OBJETO
guess = [(class_, probs.item(n)) for n, class_ in enumerate(encoder.classes_)]
print (guess)

# PROBABILIDADE AJUSTADA DE CATEGORIAS PARA O OBJETO INTEL
from operator import itemgetter
for cat, proba in sorted(guess, key=itemgetter(1), reverse=True):
    print("")
    print(' {}: {:.4f}'.format(cat,proba))


