from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
import numpy as np

lines = open("../../data/train-pos.txt").read().split('\n') + open("../../data/train-neg.txt").read().split('\n')

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(lines)
idf = vectorizer.idf_
mean_idf = np.mean(idf)
idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

pickle.dump((mean_idf, idf_dict), open("tfidf.data","wb"))

