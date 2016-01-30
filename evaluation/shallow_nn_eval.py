import cPickle as pickle
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 125
nb_classes = 2
nb_epoch = 20

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder")
args = parser.parse_args()

train_arrays, train_labels = pickle.load(open(args.folder + "train.data"))
test_arrays, test_labels = pickle.load(open(args.folder + "test.data"))
original_test_labels = test_labels
train_labels, test_labels = np_utils.to_categorical(train_labels, 2), np_utils.to_categorical(test_labels, nb_classes)

model = Sequential()
model.add(Dense(50, input_shape=(100,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


rms = RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms)

model.fit(train_arrays, train_labels, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(test_arrays, test_labels))

predictions = model.predict(test_arrays)[:,1].reshape(25000, 1)
roc_auc = roc_auc_score(original_test_labels, predictions)
print "scored: %f" % roc_auc

fpr, tpr, _ = roc_curve(original_test_labels, predictions)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.6f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Sentiment analysis ROC AUC')
plt.legend(loc="lower right")
plt.show()
