from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv('winequality-data.csv', skiprows=[0], header=None)

# select
y = df.iloc[:, 11].values
# sex definitely has the biggest effect on accuracy.
X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_val)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_val_std = sc.transform(X_val)

from sklearn.linear_model import SGDClassifier
#validation set fitting and results
sgd = SGDClassifier(n_iter=10, eta0=0.5)
sgd.fit(X_train_std, y_train)
y_pred = sgd.predict(X_val_std)

from sklearn.metrics import accuracy_score


print('Accuracy on val set with sgd: %.2f' % accuracy_score(y_val, y_pred))


model = LogisticRegression(C=1, random_state=1, solver='newton-cg', multi_class='ovr')
model.fit(X_train_std, y_train)
y_pred = model.predict(X_val_std)
print('Accuracy on val set with logistic regression: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.tree import DecisionTreeClassifier
modeltree = DecisionTreeClassifier()
modeltree.fit(X_train_std, y_train)
y_pred = modeltree.predict(X_val_std)
print('Accuracy on val set with decision tree: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.svm import SVC

modelsvc = SVC(C=.1, random_state=1)
modelsvc.fit(X_train_std, y_train)
y_pred = modelsvc.predict(X_val_std)
print('Accuracy on val set with svc: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.neighbors import KNeighborsClassifier

modelknn = KNeighborsClassifier(n_neighbors=7)
modelknn.fit(X_train_std, y_train)
y_pred = modelknn.predict(X_val_std)
print('Accuracy on val set with knn: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.neural_network import MLPClassifier

modelmlp = MLPClassifier(solver='adam', max_iter=250)
modelmlp.fit(X_train_std, y_train)
y_pred = modelmlp.predict(X_val_std)
print('Accuracy on val set with mlp: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.ensemble import GradientBoostingClassifier

modelgboost = GradientBoostingClassifier()
modelgboost.fit(X_train_std, y_train)
y_pred = modelgboost.predict(X_val_std)
print('Accuracy on val set with gradient boost: %.2f' % accuracy_score(y_val, y_pred))

#The Random forest classifier is conistently performing the best, let's use that and see if we can increase it's performance a bit more.
from sklearn.ensemble import RandomForestClassifier
param_grid = [
  {'n_estimators': [1, 10, 100, 500], 'max_features': ['auto','sqrt'], 'min_samples_leaf': [1, 10, 50, 100], 'class_weight': ['balanced', 'balanced_subsample']},
 ]

modelrantree = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid)
modelrantree.fit(X_train_std, y_train)
y_pred = modelrantree.predict(X_val_std)
print(modelrantree.best_params_)
print('Misclassified samples on val set: %d' % (y_test != y_pred).sum())
print('Misclassified samples on val set: %d' % (y_test == y_pred).sum())
print('Accuracy on val set with random forest ensemble: %.2f' % accuracy_score(y_val, y_pred))

from sklearn.ensemble import RandomForestClassifier

modelrantree = RandomForestClassifier(max_features='sqrt', n_estimators=150, min_samples_leaf=1, class_weight='balanced')
modelrantree.fit(X_train_std, y_train)
y_pred = modelrantree.predict(X_test_std)
print('Misclassified samples on test set: %d' % (y_test != y_pred).sum())
print('Misclassified samples on test set: %d' % (y_test == y_pred).sum())
print('Accuracy on test set with random forest ensemble: %.2f' % accuracy_score(y_test, y_pred))

# Scatterplot Matrix
# Relationship between quality & other features
dfsm = pd.DataFrame()
scatter_matrix(df. )
plt.show()

