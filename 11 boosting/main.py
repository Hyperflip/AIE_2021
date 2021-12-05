import math as mt
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


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


def calc_gini_single(samples):
    counts = {}
    total = 0
    for sample in samples:
        if sample not in counts:
            counts[sample] = 1
            total += 1
        else:
            counts[sample] += 1
            total += 1

    confidences = {}
    gini = 1
    for target in counts.keys():
        confidence = counts.get(target) / total
        confidences[target] = confidence

        gini -= confidence ** 2

    return gini, total, confidences

def calc_gini_for_stump(data, stump):
    smaller_than_pivot = data.loc[data[stump.col] <= stump.pivot]['target']
    larger_than_pivot = data.loc[data[stump.col] > stump.pivot]['target']

    gini_smaller, total_smaller, confidences_smaller = calc_gini_single(smaller_than_pivot)
    gini_larger, total_larger, confidences_larger = calc_gini_single(larger_than_pivot)

    total_total = total_smaller + total_larger
    gini_total = gini_smaller * (total_smaller / total_total) + gini_larger * (total_larger / total_total)

    stump.confidences['smaller'] = confidences_smaller
    stump.confidences['larger'] = confidences_larger
    stump.impurity = gini_total

    return stump


def bootstrap_weighted(data):
    data_new = pd.DataFrame().reindex_like(data).dropna()
    for _ in range(len(data['target'])):
        rand = random.uniform(0, 1)
        weight_sum = 0
        for row in data.iterrows():
            if rand < weight_sum + row[1]['sample_weight']:
                data_new = data_new.append(row[1])
                break
            else:
                weight_sum += row[1]['sample_weight']

    return data_new


if __name__ == '__main__':
    loaded_data = load_wine()
    # put data into dataframe with additional "sample weights" column
    data = pd.DataFrame(data=np.c_[loaded_data['data'],
                                   loaded_data['target'],
                                   # append np array filled with 1/n_samples as initial sample weights
                                   np.full((len(loaded_data['target']), 1), 1 / len(loaded_data['target']))],
                        columns=loaded_data['feature_names'] + ['target', 'sample_weight'])

    data_train, data_test = train_test_split(data, test_size=0.3)

    n_iterations = 10

    final_stumps = []
    for _ in range(n_iterations):
        # find best stump by column with pivot value
        possible_stumps = []
        for col in data_train.iloc[:, :-2].columns:
            # calculate pivot values for possible stumps in this col
            possible_stumps_per_col = []
            colSorted = sorted(data_train[col])
            for x_cur, x_next in zip(colSorted, colSorted[1:]):
                possible_stumps_per_col.append(
                    Stump(col=col,
                          pivot=(x_cur + x_next) / 2)
                )

            # calculate gini impurities for each stump
            for stump in possible_stumps_per_col:
                stump = calc_gini_for_stump(data_train, stump)

            # append best stump (lowest impurity) to possible_stumps over all columns
            possible_stumps.append(min(possible_stumps_per_col, key=lambda x: x.impurity))

        # choose best overall stump of iteration
        chosen_stump = min(possible_stumps, key=lambda x: x.impurity)
        final_stumps.append(chosen_stump)

        # calculate influence of stump and adjust sample weights
        incorrect = 0
        incorrect_indices = []

        for index, row in enumerate(data_train.iterrows()):
            y = row[1]['target']
            y_pred = chosen_stump.predict(row[1])
            if y_pred != y:
                incorrect += 1
                incorrect_indices.append(index)

        total_error = incorrect / len(data_train['target'])
        # adjust incorrectly classified sample weights
        for index in incorrect_indices:
            weight = data_train.iloc[index, :]['sample_weight']
            data_train.iloc[index, :]['sample_weight'] = 0.5 * mt.log((1 - total_error) / total_error)

        # normalize weights
        weight_sum = 0
        for row in data_train.iterrows():
            weight_sum += row[1]['sample_weight']
        for i in range(len(data_train['target'])):
            data_train.iloc[i, :]['sample_weight'] = data_train.iloc[i, :]['sample_weight'] / weight_sum

        # fill data anew by weighted bootstrapping
        data_train = bootstrap_weighted(data_train)

        # reset sample weights
        for i in range(len(data_train['target'])):
            data_train.iloc[i, :]['sample_weight'] = 1 / len(data_train['target'])

    # TODO predict and compare with other libraries