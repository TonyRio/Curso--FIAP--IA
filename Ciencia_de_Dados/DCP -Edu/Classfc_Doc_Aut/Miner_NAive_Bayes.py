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
print(len(twenty_train.data))