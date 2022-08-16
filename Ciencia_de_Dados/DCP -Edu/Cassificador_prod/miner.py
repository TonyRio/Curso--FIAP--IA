import gzip
import json


#Carregar Data Set

corpus = list()
with gzip.open('ecommerce.json.gz') as fp:
    for line in fp:
        entry =line.decode('utf8')
        corpus.append(json.loads(entry))
print(corpus[0]['name'])