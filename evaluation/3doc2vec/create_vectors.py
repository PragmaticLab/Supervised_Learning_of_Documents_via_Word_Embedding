from gensim.models import Doc2Vec
from sklearn.utils import shuffle
import numpy
import cPickle as pickle

model = Doc2Vec.load("../../model/d2v_50iter_5window_2mincount.mod")

train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)
test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)


for i in range(12500):
	prefix_train_pos = 'TRAIN_POS_' + str(i)
	prefix_train_neg = 'TRAIN_NEG_' + str(i)
	train_arrays[i] = model.docvecs[prefix_train_pos]
	train_arrays[i + 12500] = model.docvecs[prefix_train_neg]
	train_labels[i] = 1
	train_labels[i + 12500] = 0

	prefix_test_pos = 'TEST_POS_' + str(i)
	prefix_test_neg = 'TEST_NEG_' + str(i)
	test_arrays[i] = model.docvecs[prefix_test_pos]
	test_arrays[i + 12500] = model.docvecs[prefix_test_neg]
	test_labels[i] = 1
	test_labels[i + 12500] = 0

	if i % 1 == 0:
		print "curr at: %d / 12500" % i

train_arrays, train_labels = shuffle(train_arrays, train_labels, random_state=1337)
test_arrays, test_labels = shuffle(test_arrays, test_labels, random_state=1337)

pickle.dump((train_arrays, train_labels), open("train.data","wb"))
pickle.dump((test_arrays, test_labels), open("test.data","wb"))
