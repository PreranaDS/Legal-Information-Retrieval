import nltk

import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('wordnet')

##query=input("enter query")
##query=query.lower()
##tokens=nltk.word_tokenize(query)
##
##
##from nltk.corpus import stopwords
##from string import punctuation
###removing stop words and punctuation
##tokens_nopunct=[]
##for token in tokens:
##  if token not in stopwords.words('english') and token.isalpha():
##    tokens_nopunct.append(token)

dataframe=pd.read_csv('./SIH-Excel.csv');
queries=dataframe['case name']
q=[]
x=[]
y=[]
for qu in queries:
  y.append(qu.lower())
vocab=set()
for query in queries:
    q_words=query.lower().split(' ')
    q.append([w for w in q_words if w != ''])
    vocab=vocab.union(set(q_words))
print(q)

firstwordmatch=[]
###Preparing training set
x=input("Enter query")
if len(x.split(' '))==1:
    for qu in y:
        score=0
        if x==qu[:len(x)]:
            score=1
        firstwordmatch.append(score)
           
##One hot encoding
vocab=list(vocab)
print(len(x),len(y),'vocab length:',len(vocab))
x_train=np.zeros((1,len(vocab)))

for j in range(len(vocab)):
    if vocab[j] in x:
        x_train[0][j]=1

y_train=np.zeros((len(y),len(vocab)))
for i in range(len(y)):
    for j in range(len(vocab)):
        if vocab[j] in y[i]:
           y_train[i][j]=1
           
          

print(x_train.shape,y_train.shape)
print(x)
freq=dataframe['frequency of querying']
from scipy.spatial.distance import cosine
tf_idf={}
for i in range(y_train.shape[0]):
    if firstwordmatch!=[]:
        val=sum(x_train[0]*y_train[i])+firstwordmatch[i]
    else:
        val=sum(x_train[0]*y_train[i])
    if val!=0:
        tf_idf[i]=val

l=sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:4]#Fetch only the first four 
#Add frequency of querying
q_score={}
for i in l:
  q_score[int(i[0])]=i[1]+(freq[i[0]]*0.5)

print('')
print('#################')
print("Top four queries:")
for q in q_score:
  print(queries[q])
