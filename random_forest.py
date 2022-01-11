from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils



data = utils.get_large_train()

# print(data)

X = np.array([x for x in data.embedding.values])
y = np.array(list(data.pred.values))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=11)

rfc = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)
# rfc = DecisionTreeClassifier(criterion='entropy', random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
metric = dict()
metric['accuracy'] = accuracy_score(y_test, y_pred)
metric['recall'] = recall_score(y_test, y_pred)
metric['precision'] = precision_score(y_test, y_pred)
metric['f1'] = f1_score(y_test, y_pred)

print(metric)

y_pred_proba = rfc.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr, lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
