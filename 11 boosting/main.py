from sklearn.datasets import load_digits, load_wine
import pandas as pd
import numpy as np


class Stump:
    def __init__(self, col, pivot):
        self.col = col
        self.pivot = pivot
        self.impurity = None
        self.confidences = {'smaller': None, 'larger': None}
        self.n_incorrect = None
        self.influence = None


def calc_gini_single(data, samples):
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

    gini_smaller, total_smaller, confidences_smaller = calc_gini_single(data, smaller_than_pivot)
    gini_larger, total_larger, confidences_larger = calc_gini_single(data, larger_than_pivot)

    total_total = total_smaller + total_larger
    gini_total = gini_smaller * (total_smaller / total_total) + gini_larger * (total_larger / total_total)

    stump.confidences['smaller'] = confidences_smaller
    stump.confidences['larger'] = confidences_larger
    stump.impurity = gini_total

    return stump


if __name__ == '__main__':
    loaded_data = load_wine()
    # put data into dataframe with additional "sample weights" column
    data = pd.DataFrame(data=np.c_[loaded_data['data'],
                                   loaded_data['target'],
                                   # append np array filled with 1/n_samples as initial sample weights
                                   np.full((len(loaded_data['target']), 1), 1 / len(loaded_data['target']))],
                        columns=loaded_data['feature_names'] + ['target', 'sample_weights'])

    print(data)

    # find best stump by column with pivot value
    possible_stumps = []
    for col in data.iloc[:, :-2].columns:
        # calculate pivot values for possible stumps in this col
        possible_stumps_per_col = []
        colSorted = sorted(data[col])
        for x_cur, x_next in zip(colSorted, colSorted[1:]):
            possible_stumps_per_col.append(
                Stump(col=col,
                      pivot=(x_cur + x_next) / 2)
            )

        # calculate gini impurities for each stump
        for stump in possible_stumps_per_col:
            stump = calc_gini_for_stump(data, stump)

        # append best stump (lowest impurity) to possible_stumps over all columns
        possible_stumps.append(min(possible_stumps_per_col, key=lambda x: x.impurity))

    print('')