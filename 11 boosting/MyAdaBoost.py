from collections import Counter
import numpy as np
import pandas as pd
import random
import math as mt

class Stump:
    def __init__(self, col, pivot):
        self.col = col
        self.pivot = pivot
        self.impurity = None
        self.confidences = {'smaller': {}, 'larger': {}}
        self.n_incorrect = None
        self.influence = None

    def predict(self, x):
        confidences = self.confidences.get('smaller') if x[self.col] <= self.pivot else self.confidences.get('larger')
        return max(confidences, key=confidences.get)

def calc_gini_single(targets):
    counts = {}
    total = 0
    for target in targets:
        if target not in counts:
            counts[target] = 1
            total += 1
        else:
            counts[target] += 1
            total += 1

    confidences = {}
    gini = 1
    for target in counts.keys():
        confidence = counts.get(target) / total
        confidences[target] = confidence

        gini -= confidence ** 2

    return gini, total, confidences

def calc_gini_for_stump(X, y, stump):
    smaller_indices = np.asarray(X[:, stump.col] <= stump.pivot).nonzero()
    smaller_targets = y[smaller_indices]
    larger_indices = np.asarray(X[:, stump.col] > stump.pivot).nonzero()
    larger_targets = y[larger_indices]

    gini_smaller, total_smaller, confidences_smaller = calc_gini_single(smaller_targets)
    gini_larger, total_larger, confidences_larger = calc_gini_single(larger_targets)

    total_total = total_smaller + total_larger
    gini_total = gini_smaller * (total_smaller / total_total) + gini_larger * (total_larger / total_total)

    stump.confidences['smaller'] = confidences_smaller
    stump.confidences['larger'] = confidences_larger
    stump.impurity = gini_total

    return stump

def bootstrap_weighted(X, y, sample_weights):
    X_new = np.empty(shape=X.shape)
    X_new[:] = np.NaN
    y_new = np.empty(shape=y.shape)
    y_new[:] = np.NaN
    index = 0
    while index != X_new.shape[0]:
        rand = random.uniform(0, 1)
        weight_sum = 0
        for i in range(X.shape[0]):
            if rand < weight_sum + sample_weights[i]:
                X_new[index] = np.array(X[i, :])
                y_new[index] = np.array(y[i])
                index += 1
                break
            else:
                weight_sum += sample_weights[i]

    return X_new, y_new

class MyAdaBoost:
    def __init__(self, n_stumps):
        self.n_stumps = n_stumps
        self.stumps = []
        self.X = None
        self.y = None
        self.sample_weights =[]

    def fit(self, X, y):
        self.X = X
        self.y = y
        # set initial sample weights
        for i in range(self.X.shape[0]):
            self.sample_weights.append(1 / self.X.shape[0])

        for _ in range(self.n_stumps):
            #print('stump ' + str(_ + 1) + ' of ' + str(self.n_stumps))
            # find best stump by column with pivot value
            possible_stumps = []
            for col in range(self.X.shape[1]):
                # calculate pivot values for possible stumps in this col
                possible_stumps_per_col = []
                colSorted = sorted(self.X[:, col])
                for x_cur, x_next in zip(colSorted, colSorted[1:]):
                    possible_stumps_per_col.append(
                        Stump(col=col,
                              pivot=(x_cur + x_next) / 2)
                    )

                # calculate gini impurities for each stump
                for i in range(len(possible_stumps_per_col)):
                    possible_stumps_per_col[i] = calc_gini_for_stump(self.X, self.y, possible_stumps_per_col[i])

                # append best stump (lowest impurity) to possible_stumps over all columns
                possible_stumps.append(min(possible_stumps_per_col, key=lambda x: x.impurity))

            # choose best overall stump of iteration
            chosen_stump = min(possible_stumps, key=lambda x: x.impurity)

            # calculate influence of stump and adjust sample weights
            incorrect = 0
            incorrect_indices = []

            for i in range(self.X.shape[0]):
                target = self.y[i]
                y_pred = chosen_stump.predict(self.X[i, :])
                if y_pred != target:
                    incorrect += 1
                    incorrect_indices.append(i)

            total_error = incorrect / self.X.shape[0]
            influence = 0.5 * mt.log((1 - total_error) / total_error)
            chosen_stump.influence = influence
            # adjust incorrectly classified sample weights
            for i in range(self.X.shape[0]):
                if i in incorrect_indices:
                    self.sample_weights[i] = mt.e ** influence
                else:
                    self.sample_weights[i] = mt.e ** -influence

            self.stumps.append(chosen_stump)

            # normalize weights
            weight_sum = 0
            for i in range(X.shape[0]):
                weight_sum += self.sample_weights[i]
            for i in range(self.X.shape[0]):
                self.sample_weights[i] = self.sample_weights[i] / weight_sum

            # override data by weighted bootstrapping
            self.X, self.y = bootstrap_weighted(self.X, self.y, self.sample_weights)

            # reset sample weights
            for i in range(self.X.shape[0]):
                self.sample_weights[i] = 1 / self.X.shape[0]

    def predict(self, x):
        # majority vote prediction
        predictions = {}
        for stump in self.stumps:
            prediction = stump.predict(x)
            predictions[prediction] = predictions.get(prediction, 0) + stump.influence
        return max(predictions, key=predictions.get)
