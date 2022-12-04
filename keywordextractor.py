##with open('booktrain.txt','r',errors="ignore") as f:
##    lines=f.readlines()
##query=' '.join(lines);

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='english')
wnl = WordNetLemmatizer()
query_word2vec=[]
query_word2vec_google=[]

import pandas
import numpy as np
#dataframeold=pandas.read_csv('./SIH-Processed.csv',encoding='iso-8859-1',warn_bad_lines=True, error_bad_lines=True)
dataframe=pandas.read_csv('./SIH-Excel.csv',encoding='iso-8859-1',warn_bad_lines=True, error_bad_lines=True)
dataframe.drop(["word2vec-keywordextractor.py"], axis=1, inplace=True)
keywords=dataframe['keywords']
abstract=dataframe['abstract']
filenames=dataframe['filename']
print(filenames)
#filenames=dataframe['filename']
from nltk import pos_tag
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from string import punctuation
import gensim.downloader as api
import gensim



#training word2vec model 

all_words_vocab=[]
all_words=[]
i=0
#filter the file and write to new file
for filename in filenames:
  print(filename)
  with open("judgments/"+filename+".txt",'r',errors="ignore") as f:
    lines=f.readlines()
    filetowritestring=''
  filestring=' '.join(lines)
  filestring=filestring.lower()
  for sent in nltk.sent_tokenize(filestring):
    for word in sent.split(' '):
      for char in set(punctuation).union(set(['\n','\t','\r','(',')','[',']','â','€','”','œ'])):
        word=word.replace(char,' ')
      all_words_vocab.append(word.strip())
      filetowritestring+=' '+(wnl.lemmatize(word))
  fw=open("judgment_refined/"+filename+".txt","w")
  fw.write(filetowritestring)
  fw.close()
  i+=1


  article_text=''
  for words in lines:
    article_text+=words.lower()
  all_sentences = nltk.sent_tokenize(article_text)

  for sent in all_sentences:
    wordsinsent=[]
    for word in sent.split(' '):
      if word not in stopwords.words('english'):
        for char in set(punctuation).union(set(['\n','\t','\r','(',')','[',']'])):
          word=word.replace(char,' ')
        wordsinsent.append(wnl.lemmatize(str(word)))
    all_words.append(wordsinsent)
  
file1 = open("vocab_spelling.txt", "w",encoding='utf-8')
file1.write(','.join(all_words_vocab))
file1.close()


print("preprocessing done")
try:
  word2vec = Word2Vec.load("word2vec.model")
  word2vec.build_vocab(all_words, update = True)
  word2vec.train(all_words, compute_loss=True, epochs = 100)
  
except:
  word2vec = Word2Vec(all_words, min_count=1, vector_size=100, epochs=100)#increase the number of epochs and dimensions
  word2vec.save("word2vec.model")
##
##word2vec.get_latest_training_loss()
#PATH_TO_GOOGLE_WORD2VEC = api.load("word2vec-google-news-300", return_path=True)
#print(PATH_TO_GOOGLE_WORD2VEC)
#wvgoogle=gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
#PATH_TO_GOOGLE_WORD2VEC='C:/Users/Toshiba/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'
#wvgoogle=gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_GOOGLE_WORD2VEC, binary=True)
#google word2vec is slowing it down a lot
print("*********************\nabstract\n",abstract)
for query in abstract:
  query=query.lower()
  tokens=query.split(' ')
  print(tokens)
  print(keywords)


  #removing stop words and punctuation
  tokens_nopunct=[]
  for token in tokens:
    if token not in stopwords.words('english') and token not in punctuation and token not in ['\n','\t','\r']:
      for char in set(punctuation).union(set(['\n','\t','\r','(',')'])):
        token=token.replace(char,' ')
      tokens_nopunct.append(wnl.lemmatize(token))
        

  query_embedding = [word2vec.wv[w] for w in tokens_nopunct if w in word2vec.wv.key_to_index]
  query_word2vec.append(sum(query_embedding)/len(query_embedding))
  #query_embedding_google = [wvgoogle[w] for w in tokens_nopunct if w in wvgoogle.key_to_index]
  #query_word2vec_google.append(sum(query_embedding_google)/len(query_embedding_google))

dataframe['word2vec_embedding']=query_word2vec
#dataframe['word2vec_embedding_google']=query_word2vec_google
pandas.set_option('display.max_colwidth', None)
#dataframe['word2vec_embedding_google']=query_word2vec_google
print('done')
#dataframeconcat=pandas.concat([dataframeold, dataframe])
#print(dataframe)
#print(dataframeold)
dataframe.to_csv(r'SIH-Processed.csv')
f.close()
