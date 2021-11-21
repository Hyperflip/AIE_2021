from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import randint
from collections import Counter


class BaseLearner:
    def __init__(self, clf, features_start, features_end):
        self.clf = clf
        self.features_start = features_start
        self.features_end = features_end


# hard-vote ensemble prediction
def ensemble_predict_single(base_learners, x):
    results = []
    for bl in base_learners:
        # get features that base learner supports
        x_adj = x[bl.features_start:bl.features_end].reshape(1, -1)
        results.append((bl.clf.predict(x_adj))[0])
    return Counter(results).most_common(1)[0][0]


# data loading and splitting
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# fit base learners by random patching
n_base_learners = 20

max_samples = int(len(X_train) / n_base_learners)
min_samples = int(max_samples * 0.5)
min_features = int(len(X_train[0]) * 0.1)

base_learners = []
for i in range(n_base_learners):
    # get amount of samples to pick
    n_samples = randint(min_samples, max_samples)
    # get starting index of samples (adjust for unimplemented wrap-around)
    samples_start = randint(0, len(X_train) - n_samples)
    samples_end = samples_start + n_samples

    # same for features
    n_features = randint(min_features, len(X_train[0]))
    features_start = randint(0, len(X_train[0]) - n_features)
    features_end = features_start + n_features

    # pick data
    X_train_rand = X_train[samples_start:samples_end, features_start:features_end]
    y_train_rand = y_train[samples_start:samples_end]

    # fit clf and append to base learners
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train_rand, y_train_rand)
    bl = BaseLearner(clf, features_start, features_end)
    base_learners.append(bl)

# train single classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# test with all X_test
correct_ensemble = 0
correct_single = 0
for x, y in zip(X_test, y_test):
    if y == ensemble_predict_single(base_learners, x):
        correct_ensemble += 1
    if y == clf.predict(x.reshape(1, -1)):
        correct_single += 1
accuracy_ensemble = correct_ensemble / len(X_test)
accuracy_single = correct_single / len(X_test)

print('accuracy of ensemble: ' + str(accuracy_ensemble))
print('accuracy of single classifier: ' + str(accuracy_single))
