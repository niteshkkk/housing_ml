import numpy as np
import pandas as pd
import sys
from zlib import crc32
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from ml.download_data.load_housing_data import HousingData
from ml.test_set.split_test_set import GenerateTrainingSet
import mpld3_graphs.utils as utils
from ml.graph_analysis.base_analysis import BaseAnalysis
from ml.graph_analysis.analysis_plots import Analysis

np.random.seed(42)
test_ratio = 0.2
random_state = 42


class Cleaner(object):

    def __init__(self, strat_train_set, strat_test_set):
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

    def clean(self):
        self.housing = strat_train_set.drop("median_house_value", axis=1)
        self.housing_label = strat_train_set["median_house_value"].copy()
        housing = self.housing
        housing_label = self.housing_label
        print(housing.info())
        sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head(20)
        print(sample_incomplete_rows)
        print(housing[housing.isnull().any(axis=1)].head(30))
        print(housing.loc[[4629, 6068]])


if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'
    analysis = Analysis(HousingData()).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()
    print("\n\n --------- Cleaning ----------- \n\n")
    cleaner = Cleaner(strat_train_set, strat_test_set)
    cleaner.clean()

    if show:
        plt.show()
