from gensim.models import Word2Vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

train_data = open('../data/train-pos.txt').read().split('\n')[:12500] + open('../data/train-neg.txt').read().split('\n')[:12500]
train_data = [line.split() for line in train_data]

model = Word2Vec(size=100, alpha=0.025, window=5, min_count=2, seed=1337, workers=8)

model.build_vocab(train_data)

iter_count = 50
for i in range(50):
	print "on iter %d" % i
	model.train(train_data)

model.save('../model/w2v_50iter_5window_2mincount.mod')
