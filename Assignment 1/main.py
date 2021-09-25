from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import time

def print_metrics(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    # calculate MSE, R2 and p-value
    n = len(X)
    predictions = reg.predict(X)
    # residual sum of squares
    rss = np.sum((predictions - y) ** 2)
    tss = np.sum((np.mean(y) - y) ** 2)
    mse = rss / n
    r_2 = 1 - (rss / tss)
    # p-value
    _, p_values = f_regression(X, y)

    print('MSE:', mse)
    print('R2:', r_2)
    print('average p-value:', np.mean(p_values))

def backward_selection(data):
    while True:
        p_values = f_regression(data.iloc[:, :-1], data.iloc[:, -1])[1]

        if (p_values <= 0.05).all() or data.shape[1] == 2:
            break

        # eliminate feature(s) with highest p-value
        bad_index = p_values.argmax(axis=0)
        data = data.drop(data.columns[bad_index], axis=1)

    return data

def forward_selection(data):
    reg = LinearRegression()
    reg.fit(data.iloc[:, :-1], data.iloc[:, -1])
    intercept = reg.intercept_

    # features that haven't been integrated or discarded yet
    untested_features = data.iloc[:, :-1]
    # null model which will be expanded
    model = pd.DataFrame(data=data.iloc[:, -1])
    run = True
    while run:
        rss_list = []
        for feature in untested_features.columns:
            temp_model = model
            temp_model.insert(0, column=feature, value=data[feature])

            # discard feature if it results in a bad p-value
            p_values = f_regression(temp_model.iloc[:, :-1], temp_model.iloc[:, -1])[1]
            if (p_values <= 0.05).any():
                # insert bad rss to keep indices in order
                rss_list.append(1000000)
                continue

            # calculate rss for temp model
            reg.fit(temp_model.iloc[:, :-1], temp_model.iloc[:, -1])
            predictions = reg.predict(temp_model.iloc[:, :-1])
            rss_list.append(np.sum((predictions - y) ** 2))

        # find feature with smallest rss and insert into null model
        min_index = np.argmin(rss_list)
        # ignore bad features that have been flagged with high rss'
        if rss_list[min_index] != 1000000:
            model.insert(0, column=min_index, value=data.iloc[:, min_index])
        # drop from untested features
        untested_features = untested_features.drop(untested_features.columns[min_index], axis=1)

        # break if all have been tested
        if untested_features.shape[1] == 0:
            break

    return model

if __name__ == '__main__':
    n_features = 100
    X, y = make_regression(n_samples=100, n_features=n_features, noise=25)
    data = pd.DataFrame(data=np.c_[X, y])

    # this doesn't work, but at least the code can be discussed
    #print(data.shape[1])
    #data1 = forward_selection(data)
    #print(data1.shape[1])

    print('before backward selection')
    print('feature count:', data.shape[1])
    print_metrics(data.iloc[:, :-1], data.iloc[:, -1])

    data2 = backward_selection(data)

    print('after backward selection')
    print('feature count:', data2.shape[1])
    print_metrics(data2.iloc[:, :-1], data2.iloc[:, -1])