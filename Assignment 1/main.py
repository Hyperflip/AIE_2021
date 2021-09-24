from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

def backward_selection(data):
    while True:
        p_values = f_regression(data.iloc[:, :-1], data.iloc[:, -1])[1]
        if (p_values <= 0.05).all() or data.shape[1] == 2:
            break
        # eliminate feature(s) with highest p-value
        bad_index = p_values.argmax(axis=0)
        data = data.drop(data.columns[bad_index], axis=1)

    return data

if __name__ == '__main__':
    n_features = 100
    X, y = make_regression(n_samples=100, n_features=n_features, noise=25)
    data = pd.DataFrame(data=np.c_[X, y])

    print(data.shape[1])

    data = backward_selection(data)

    print(data.shape[1])

    exit(0)


    features = []
    for i in range(0, n_features):
        features.append(X[:, i])

    _, p_value = f_regression(X, y)
    print(p_value)

    for feature in features:
       _, p_value = f_regression(feature.reshape(-1, 1), y)
       print(p_value)


    #reg = LinearRegression()
    #reg.fit(X, y)

    #_, p_value = f_regression(X, y)
    #print(p_value)

    #print(reg.coef_)
    #print(reg.intercept_)
