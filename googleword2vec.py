import gensim.downloader as api
import gensim
path = api.load("word2vec-google-news-300", return_path=True)
print(path)
wvgoogle=gensim.models.KeyedVectors.load_word2vec_format('C:/Users/JNCPG/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)
print(wvgoogle['cat'])
