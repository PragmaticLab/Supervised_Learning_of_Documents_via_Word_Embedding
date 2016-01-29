import xgboost as xgb
import cPickle as pickle
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder")
args = parser.parse_args()

train_arrays, train_labels = pickle.load(open(args.folder + "train.data"))
test_arrays, test_labels = pickle.load(open(args.folder + "test.data"))
dtrain = xgb.DMatrix(train_arrays, label=train_labels)
dtest = xgb.DMatrix(test_arrays, label=test_labels)

print "starting training"
num_rounds = 2500
param = {'max_depth': 5, 'eta':0.04, 'silent':1, 'objective':'binary:logistic', 'nthread': 4, 'eval_metric': 'auc', 'seed': 1337, 'lambda': 1.15, 'colsample_bytree': 0.3}
evallist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param.items(), dtrain, num_rounds, evallist, early_stopping_rounds=50)
predictions = bst.predict(xgb.DMatrix(train_arrays))

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
