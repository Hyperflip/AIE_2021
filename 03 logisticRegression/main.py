import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# data loading
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, n_samples=100)

# normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# initialize models
classifiers = {
    'knn': KNeighborsClassifier(),
    'tree': DecisionTreeClassifier(),
    'logReg': LogisticRegression()
}

# split into sets
n_splits = 10
per_split_accuracies = {'knn': [], 'tree': [], 'logReg': []}
for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / n_splits, random_state=42 + i)

    # fit models
    classifiers['knn'].fit(X_train, y_train)
    classifiers['tree'].fit(X_train, y_train)
    classifiers['logReg'].fit(X_train, y_train)

    # predict with each model
    predictions = {
        'knn': classifiers['knn'].predict(X_test),
        'tree': classifiers['tree'].predict(X_test),
        'logReg': classifiers['logReg'].predict(X_test)
    }

    # calculate accuracies
    per_split_accuracies['knn'].append(np.sum(predictions['knn'] == y_test) / len(y_test))
    per_split_accuracies['tree'].append(np.sum(predictions['tree'] == y_test) / len(y_test))
    per_split_accuracies['logReg'].append(np.sum(predictions['logReg'] == y_test) / len(y_test))

avg_accuracies = {
    'knn': round(np.sum(per_split_accuracies['knn']) / n_splits, 2),
    'tree': round(np.sum(per_split_accuracies['tree']) / n_splits, 2),
    'logReg': round(np.sum(per_split_accuracies['logReg']) / n_splits, 2)
}

print(per_split_accuracies)
print(avg_accuracies)
