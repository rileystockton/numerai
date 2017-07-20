from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
np.random.seed(1)

# Get data
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]

# Create models
model = Pipeline([('svc', SVC(kernel='linear', degree=1, probability=True))])
boosted_model = AdaBoostClassifier(base_estimator = model)

# Fit to training data
model.fit(X_train, y_train)
boosted_model.fit(X_train, y_train)

# Score on testing data
model_score = model.score(X_test, y_test)
boosted_model_score = boosted_model.score(X_test, y_test)

print 'Without boosting: %f' % model_score
print 'With boosting: %f' % boosted_model_score
