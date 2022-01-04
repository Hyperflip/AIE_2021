from seaborn import load_dataset
from sklearn.datasets import load_diabetes, make_regression
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':

    data_tips = load_dataset('tips')
    data_tips['tip'] = data_tips.pop('tip')     # move target col to end
    # drop non-numeric features
    data_tips = data_tips.drop(columns='sex')
    data_tips = data_tips.drop(columns='smoker')
    data_tips = data_tips.drop(columns='day')
    data_tips = data_tips.drop(columns='time')

    data_mpg = load_dataset('mpg')
    data_mpg['mpg'] = data_mpg.pop('mpg')       # move target col to end
    data_mpg = data_mpg.drop(columns='origin')
    data_mpg = data_mpg.drop(columns='name')
    data_mpg = data_mpg.dropna()

    diabetes = load_diabetes()
    data_diabetes = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                                 columns=diabetes['feature_names'] + ['target'])

    X1, y1 = make_regression(n_samples=500, n_features=30, n_informative=10, noise=25)
    X2, y2 = make_regression(n_samples=250, n_features=20, n_informative=10, noise=10)
    X3, y3 = make_regression(n_samples=250, n_features=15, n_informative=5, noise=30)

    datasets = {
        'tips': (data_tips.iloc[:, :-1].to_numpy(), data_tips.iloc[:, -1].to_numpy()),
        'mpg': (data_mpg.iloc[:, :-1].to_numpy(), data_mpg.iloc[:, -1].to_numpy()),
        'diabetes': (data_diabetes.iloc[:, :-1].to_numpy(), data_diabetes.iloc[:, -1].to_numpy()),
        'artificial1': (X1, y1),
        'artificial2': (X2, y2),
        'artificial3': (X3, y3)
    }

    regressors = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'kNN': KNeighborsRegressor(n_neighbors=5),
        'svr': SVR(),
        'ensemble': RandomForestRegressor(n_estimators=10),
        'ann1': MLPRegressor(hidden_layer_sizes=100, activation='relu'),
        'ann2': MLPRegressor(hidden_layer_sizes=75, activation='relu'),
        'ann3': MLPRegressor(hidden_layer_sizes=50, activation='relu'),
        'ann4': MLPRegressor(hidden_layer_sizes=50, activation='identity'),
        'ann5': MLPRegressor(hidden_layer_sizes=50, activation='logistic')
    }

    n_splits = 10

    # loop over datasets
    per_data_rmsds = {}
    for data_name in datasets.keys():
        X, y = datasets[data_name]

        per_reg_rmsds = {}
        for reg_name in regressors.keys():

            per_split_rmsd = []
            for _ in range(n_splits):
                # split into train and test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                # fit regressor
                regressors[reg_name].fit(X_train, y_train)

                # benchmark regressors
                squared_dev_sum = 0
                for _x, _y in zip(X_test, y_test):
                    y_hat = regressors[reg_name].predict(_x.reshape(1, -1))
                    squared_dev_sum += abs(y_hat[0] - _y) ** 2
                mean_squared_dev = squared_dev_sum / X_test.shape[0]
                rmsd = mean_squared_dev ** 0.5
                per_split_rmsd.append(rmsd)

            per_reg_rmsds[reg_name] = per_split_rmsd

        per_data_rmsds[data_name] = per_reg_rmsds






