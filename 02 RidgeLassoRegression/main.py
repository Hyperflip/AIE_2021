from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from MyRidgeRegression import MyRidgeRegression
from sklearn.linear_model import Ridge

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

myRidgeRegressor = MyRidgeRegression(1)
ridgeRegressor = Ridge(max_iter=1000, alpha=1)

myRidgeRegressor.fit(X_train, y_train)
ridgeRegressor.fit(X_train, y_train)

mySum = 0
sum = 0
for X_test, y_test in zip(X_test, y_test):
    myY_pred = myRidgeRegressor.predict(X_test)
    mySum += (y_test - myY_pred) ** 2

    y_pred = ridgeRegressor.predict(X_test)
    sum += (y_test - y_pred) ** 2

myRss = mySum ** 0.5
rss = sum ** 0.5

print(myRss, rss)