from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import cPickle as pickle
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder")
args = parser.parse_args()

train_arrays, train_labels = pickle.load(open(args.folder + "train.data"))
test_arrays, test_labels = pickle.load(open(args.folder + "test.data"))

print "starting training"
# Logistic
classifier = LogisticRegression(n_jobs=1, verbose=3, random_state=1337, class_weight='balanced', C=0.7)
classifier.fit(train_arrays, train_labels)
predictions = classifier.predict_proba(test_arrays)[:,1]
logistic_roc_auc = roc_auc_score(test_labels, predictions)
logistic_fpr, logistic_tpr, _ = roc_curve(test_labels, predictions)

#RF
rf = RandomForestRegressor(n_estimators=20, random_state=1337, verbose=3, n_jobs=4, max_features=0.5)
rf = rf.fit(train_arrays, train_labels)
predictions = rf.predict(test_arrays)
rf_roc_auc = roc_auc_score(test_labels, predictions)
rf_fpr, rf_tpr, _ = roc_curve(test_labels, predictions)

#NN
batch_size = 125
nb_classes = 2
nb_epoch = 20
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
nn_roc_auc = roc_auc_score(original_test_labels, predictions)
nn_fpr, nn_tpr, _ = roc_curve(original_test_labels, predictions)


plt.figure()
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression ROC curve (area = %0.6f)' % logistic_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest ROC curve (area = %0.6f)' % rf_roc_auc)
plt.plot(nn_fpr, nn_tpr, label='Shallow Neural Network ROC curve (area = %0.6f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Sentiment analysis ROC AUC for %s' % args.folder[1:len(args.folder) - 1])
plt.legend(loc="lower right")
plt.show()
