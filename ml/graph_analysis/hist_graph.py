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

np.random.seed(42)
test_ratio = 0.2
random_state = 42


class Analysis(BaseAnalysis):

    def __init__(self, data_wrapper, show=False):
        self.data_wrapper = data_wrapper
        self.data_frame = self.data_wrapper.get_data_frame()
        super().__init__(show)

    def generate_train_set(self):
        training_set = GenerateTrainingSet(self.data_frame)
        self.train_set, self.test_set = training_set.split_train_set_sklearn(
            test_ratio, random_state)

    def base_graph(self):
        housing = self.data_frame

        super().plot_and_save_hist(housing, "median_income",
                                   "housing_median_income")
        # housing["median_income"].hist()
        # utils.save_as_html("housing_median_income")

        print(housing["median_income"].head(20))
        print(housing["median_income"].describe())

        housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
        print(housing["income_cat"].head(40))
        # Label those above 5 as 5
        housing["income_cat"].where(
            housing["income_cat"] < 5, 5.0, inplace=True)

        print(housing["income_cat"].value_counts())
        housing["income_cat"].hist()
        # plt.show()
        super().plot_and_save_hist(housing, "income_cat", "housing_income_cat")
        # housing["income_cat"].hist()
        # utils.save_as_html("housing_income_cat")
        self.generate_train_set()

    def stratified_sampling(self):
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42)
        i = 0
        housing = self.data_frame

        for train_index, test_index in split.split(housing, housing["income_cat"]):
            print(" i = ", i)
            i = i + 1
            print("train_index = ", train_index)
            print("test_set = ", test_index)
            self.strat_train_set = housing.loc[train_index]
            self.strat_test_set = housing.loc[test_index]
        print("id = ", id(self.strat_test_set), " id = ", id(housing))
        print("s test = ", self.strat_test_set)

        print(" percentage of income cat in stratified test set\n",
              self.strat_test_set["income_cat"].value_counts() / len(self.strat_test_set))
        print("\npercentage of income cat in original housing data set\n",
              housing["income_cat"].value_counts() / len(housing))

    def income_cat_proportions(self, data):
        return data["income_cat"].value_counts() / len(data)

    def comparison_matrix(self):
        housing = self.data_frame
        train_set = self.train_set
        test_set = self.test_set
        strat_test_set = self.strat_test_set

        compare_props = pd.DataFrame({
            "Overall": self.income_cat_proportions(housing),
            "Stratified": self.income_cat_proportions(strat_test_set),
            "Random": self.income_cat_proportions(test_set)
        })

        print(compare_props)

        compare_props = pd.DataFrame({
            "Overall": self.income_cat_proportions(housing),
            "Stratified": self.income_cat_proportions(strat_test_set),
            "Random": self.income_cat_proportions(test_set)
        }).sort_index()

        print(compare_props)
        r = compare_props["Random"]
        o = compare_props["Overall"]
        s = compare_props["Stratified"]

        compare_props["Rand. %error"] = ((r - o) / o) * 100
        compare_props["Strat. %error"] = ((s - o) / o) * 100
        print("\n Comparison Matrix \n", compare_props)
        self.comparison_matrix = compare_props

    def drop_stratified_columns(self):
        for set_ in (self.strat_train_set, self.strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

    def get_stratified_train_test_set(self):
        return self.strat_train_set, self.strat_test_set

    def pipeline(self):
        self.base_graph()
        self.stratified_sampling()
        self.comparison_matrix()
        self.drop_stratified_columns()

        return self

if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'
    print("show = ", show)

    analysis = Analysis(HousingData(), show).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()
    print(len(strat_train_set))
