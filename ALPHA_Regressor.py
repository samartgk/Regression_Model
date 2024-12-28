#####                                                          By : ALPHA                                          #####
########################################################################################################################

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import gc
import statistics
import pandas as pd
import numpy as np
gc.collect()

class bp_model:
    def __init__(self, method='mean'):
        self.Method = method
        self.Train_data = None
        self.Important_features = None
        self.Feature_coefficients = None

    def fit(self, x, y):
        feature_list = x.columns
        coeffs = []
        for i, feat in enumerate(feature_list):
            coeff = np.polyfit(np.array(x[feat]), np.array(y), 4)
            coeffs.append(coeff)

        means = []
        for i in range(len(coeffs)):
            x_new = feature_list[i]
            p = np.poly1d(coeffs[i])
            y_new = p(np.array(x[x_new]))
            means.append(np.mean(np.abs(y - y_new)))

        zipped_lists = zip(means, feature_list, coeffs)
        sorted_pairs = sorted(zipped_lists)
        means, feature_list, coeffs = zip(*sorted_pairs)
        self.Feature_coefficients = list(coeffs)
        self.Important_features = list(feature_list)
        self.Train_data = x[self.Important_features]

    def predict(self, x):
        test_data = x[self.Important_features]
        predictions = []
        for i in range(len(self.Feature_coefficients)):
            x_new = self.Important_features[i]
            p = np.poly1d(self.Feature_coefficients[i])
            y_pred = p(np.array(test_data[x_new]))
            predictions.append(y_pred)

        if self.Method == 'median':
            transposed = list(zip(*predictions))
            medians = [statistics.median(column) for column in transposed]
            return medians
        elif self.Method == 'mode':
            transposed = list(zip(*predictions))
            modes = [statistics.mode(column) for column in transposed]
            return modes
        else:
            transposed = list(zip(*predictions))
            means = [statistics.mean(column) for column in transposed]
            return means


