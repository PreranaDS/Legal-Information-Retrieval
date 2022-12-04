import re
import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from string import punctuation


#Query input
query=input("enter query")
query=query.lower()
tokens=[]
for word in query.split(' '):
    #if word not in stopwords.words('english'):
    for char in set(punctuation).union(set(['\n','\t','\r','(',')','[',']'])):
          word=word.replace(char,' ')
    tokens.append(word)

#tf of query
vocab=list(set(tokens))
tf_query=np.zeros(len(vocab))
for i in range(len(vocab)):
    for token in tokens:
        if token==vocab[i]:
            tf_query[i]+=1

#Searching documents for word
filenames=['The State Of Maharashtra vs Rakesh Manohar Kamble','arms act','central government act','chedalavada subbayya','Commissioner Of Gift-Tax, ... vs Basant Kumar Aditya Vikram Birla on 8 July, 1981','Narayana Annavi And Two Ors. vs K. Ramalinga Annavi','Sassoon J. David & Co. (P) Ltd., ... vs C.I.T., Bombay on 3 May, 1979','Shobha Rani vs Madhukar Reddi on 12 November, 1987','Sri Raja Row Venkata Mahipathy ... vs Sri Raja Venkata Mahipathy Surya ... on 19 March, 1915']
tf=np.zeros((len(filenames),len(vocab)))
idf=np.zeros(len(vocab))
tf_idf=np.zeros((len(filenames),len(vocab)))
for i in range(len(filenames)):
    with open(filenames[i]+".txt",'r',errors="ignore") as f:
        lines=f.readlines()
    filestring=' '.join(lines)
    filestring=filestring.lower()
    for char in set(punctuation).union(set(['\n','\t','\r','(',')','[',']'])):
          filestring=filestring.replace(char,' ')
    print("\n##################\n",filenames[i],":",filestring)
    for j in range(len(vocab)):
        tf[i][j]+=len(re.findall(r'\s'+vocab[j]+'\s',filestring))
        idf[j]+=len(re.findall(r'\s'+vocab[j]+'\s',filestring))

tf_idf=tf/idf

#cosine similarity
from scipy.spatial.distance import cosine
cosinelist=[]
for ti in tf_idf:
    cosinelist.append(sum(tf_query*ti))
    #print(w,'and',w.shape,'and',query_word2vec.shape,'and',type(w),type(query_word2vec))

df_score=pd.DataFrame({'Judgment':filenames,'tf-idf_score':cosinelist})
df_score.sort_values(by=['tf-idf_score'], ascending=False, inplace=True)
print(df_score)
