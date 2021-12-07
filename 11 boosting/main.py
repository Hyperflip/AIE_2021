from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from MyAdaBoost import MyAdaBoost

if __name__ == '__main__':
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    classifiers = {}
    clfTree = DecisionTreeClassifier()
    clfTree.fit(X_train, y_train)
    classifiers['clfTree'] = clfTree
    clfForest = RandomForestClassifier()
    clfForest.fit(X_test, y_test)
    classifiers['clfForest'] = clfForest
    myAdaBoost = MyAdaBoost(n_stumps=10)
    myAdaBoost.fit(X_train, y_train)
    classifiers['myAdaBoost'] = myAdaBoost

    for key in classifiers.keys():
        correct = 0
        for _x, _y in zip(X_test, y_test):
            if key != 'myAdaBoost':
                _x = _x.reshape(1, -1)
            if classifiers[key].predict(_x) == _y:
                correct += 1
        accuracy = correct / X_test.shape[0]

        print(key + ', accuracy: ' + str(accuracy))
