from sklearn.linear_model import LogisticRegression
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
classifier = LogisticRegression(n_jobs=1, verbose=3, random_state=1337, class_weight='balanced', C=0.7)
classifier.fit(train_arrays, train_labels)
predictions = classifier.predict_proba(test_arrays)[:,1]

roc_auc = roc_auc_score(test_labels, predictions)
print "scored: %f" % roc_auc

fpr, tpr, _ = roc_curve(test_labels, predictions)

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
