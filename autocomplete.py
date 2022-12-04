#conda create -n tf tensorflow
#conda activate tf
import tensorflow as tf
import bs4 as bs
import re
import requests
import nltk
import tensorflow.keras.utils as np_utils
import numpy as np

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
    
urls=['https://www.ruf.rice.edu/~kemmer/Words04/usage/jargon_legal.html','https://www.advocatekhoj.com/library/judgments/announcement.php?WID=14838','https://www.advocatekhoj.com/library/judgments/announcement.php?WID=14837','https://www.advocatekhoj.com/library/judgments/announcement.php?WID=14835','https://www.advocatekhoj.com/library/judgments/announcement.php?WID=14834','https://www.advocatekhoj.com/library/judgments/index.php?go=1983/july/1.php',
      'https://www.advocatekhoj.com/library/judgments/index.php?go=2021/august/4.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2021/august/12.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2020/december/2.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2021/march/24.php','https://www.advocatekhoj.com/library/judgments/announcement.php?WID=14831',
      'https://www.advocatekhoj.com/library/judgments/index.php?go=2006/january/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2006/december/3.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2006/november/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2006/january/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2021/january/11.php',
      'https://www.advocatekhoj.com/library/judgments/index.php?go=1983/january/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2006/february/5.php','https://www.advocatekhoj.com/library/judgments/index.php?go=1992/september/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=1992/january/1.php','https://www.advocatekhoj.com/library/judgments/index.php?go=2021/february/51.php',
      'https://www.casemine.com/judgement/in/5609af33e4b0149711415cab','https://www.livelaw.in/top-stories/supreme-court-holds-a-major-daughter-not-entitled-to-education-expenses-from-father-as-she-does-not-want-to-maintain-relationship-with-him-194363','https://www.advocatekhoj.com/library/judgments/index.php?go=1969/january/7.php']

article_text=""
for url in urls:
    print(url)
page = requests.get(url)
ml_wiki = page.text
ml_wiki_parsed = bs.BeautifulSoup(ml_wiki,'lxml')
paragraphs = ml_wiki_parsed.find_all('p')
for p in paragraphs:
    para_text = p.text.lower()
    #cleaning the text
    #para_text = re.sub('[^a-zA-Z]', ' ', para_text)
    #para_text=re.sub(r'\s+', ' ', para_text)
    article_text+=para_text

# Preparing the dataset
all_words_sentence = nltk.sent_tokenize(article_text)
vocab={''}

X=[]
all_words_refined=[]

from nltk.corpus import stopwords
from string import punctuation
# Removing Stop Words
for sent in all_words_sentence:
    all_words_in_sentence=sent.split(' ')
    all_words_in_sentence_refined=[]
    for w in all_words_in_sentence:
        if w not in stopwords.words('english'):
            for char in set(punctuation).union(set(['\n','\t','\r','(',')','[',']'])):
                w=w.replace(char,' ')
            all_words_in_sentence_refined.append(w)
    vocab=vocab.union(set(all_words_in_sentence_refined))
    all_words_refined.append(all_words_in_sentence_refined)
  

vocab=list(vocab)


#creating n grams
n_grams=[]
labels=[]
for sent in all_words_refined:
    print(sent)
    for i in range(len(sent)-1):
        if sent[:i] !=[]:
            n_grams.append(sent[:i])
            labels.append(sent[i+1])

print(n_grams,labels)
x_train=np.zeros((len(n_grams),len(vocab),1))
y_train=np.zeros((len(n_grams),len(vocab)))
for i in range(len(n_grams)):
    for j in range(len(vocab)):
        if vocab[j] in n_grams[i]:
            x_train[i][j][0]=1
        if vocab[j] in labels[i]:
            y_train[i][j]=1   
            
        

#using ngram for lstm to predict the next word until the word predicted is '.'



def create_model():
    layers=tf.keras.layers
    models=tf.keras.models
    model = models.Sequential()
    # ----------Add Input Layer
    #model.add(layers.Input(len(vocab),10,len(vocab)))
    # ----------Add Hidden Layer 1 - LSTM Layer
    model.add(layers.LSTM(100,input_shape=(x_train.shape[1],1)))
    model.add(layers.Dropout(0.1))
    # ----------Add Output Layer
    model.add(layers.Dense(len(vocab), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())
model.fit(x_train, y_train, epochs=10, steps_per_epoch=5, verbose=5)#make epoch 100
ypred=model(x_train[60].reshape((1,len(vocab),1)),training=False)
for i in range(len(x_train[60])):
  if x_train[60][i]==1:
    print(vocab[i],end=' ')
print('\n')
words={}
for i in range(len(ypred[0])):
  words.setdefault(vocab[i],ypred[0][i])
sorteddict=dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
first_n_values = list(sorteddict.values())[:20]
first_n_keys=list(sorteddict.keys())[:20]
for i in range(len(first_n_keys)):
    print(first_n_keys[i],':',first_n_values[i])
print(x_train[5],ypred[0])
