from gensim.models import Word2Vec
from sklearn.utils import shuffle
import numpy
import cPickle as pickle

model = Word2Vec.load("../../model/w2v_50iter_5window_5mincount.mod")

vocab_list = model.vocab.keys()
vocab_dict = {}
for vocab in vocab_list:
	vocab_dict[vocab] = model[vocab]

train_pos = open("../../data/train-pos.txt").read().split('\n')
train_neg = open("../../data/train-neg.txt").read().split('\n')
test_pos = open("../../data/test-pos.txt").read().split('\n')
test_neg = open("../../data/test-neg.txt").read().split('\n')

train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)
test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

def getVectorForSentence(sentence):
	vec = numpy.zeros(100)
	count = 0
	words = sentence.split()
	for word in words:
		if word in vocab_list:
			vec += vocab_dict[word]
			count += 1
	return vec / count

for i in range(12500):
	train_pos_vector = getVectorForSentence(train_pos[i])
	train_neg_vector = getVectorForSentence(train_neg[i])
	train_arrays[i] = train_pos_vector
	train_arrays[i + 12500] = train_neg_vector
	train_labels[i] = 1
	train_labels[i + 12500] = 0

	test_pos_vector = getVectorForSentence(test_pos[i])
	test_neg_vector = getVectorForSentence(test_neg[i])
	test_arrays[i] = test_pos_vector
	test_arrays[i + 12500] = test_neg_vector
	test_labels[i] = 1
	test_labels[i + 12500] = 0

	if i % 1 == 0:
		print "curr at: %d / 12500" % i

train_arrays, train_labels = shuffle(train_arrays, train_labels, random_state=1337)
test_arrays, test_labels = shuffle(test_arrays, test_labels, random_state=1337)

pickle.dump((train_arrays, train_labels), open("train.data","wb"))
pickle.dump((test_arrays, test_labels), open("test.data","wb"))
