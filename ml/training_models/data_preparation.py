from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import sys
from zlib import crc32
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder
from ml.download_data.load_housing_data import HousingData
from ml.test_set.split_test_set import GenerateTrainingSet
import mpld3_graphs.utils as utils
from ml.graph_analysis.analysis_plots import Analysis
from encoder import CategoricalEncoder
from ml.transformer import CombinedAttributesAdder, DataFrameSelector
from sklearn.pipeline import Pipeline, FeatureUnion

np.random.seed(42)
test_ratio = 0.2
random_state = 42

cat_pipeline = None
num_pipeline = None


class DataPreparation(object):

    def __init__(self, strat_train_set, strat_test_set):
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

        self.housing = strat_train_set.drop("median_house_value", axis=1)
        self.housing_num = self.housing.drop("ocean_proximity", axis=1)
        self.housing_labels = strat_train_set["median_house_value"].copy()

        self.num_attribs = list(self.housing_num)
        self.cat_attribs = ['ocean_proximity']

    def create_numerical_pipeline(self):
        list_num_attribs = self.num_attribs
        print("List Numerical Attributes = ", list_num_attribs)
        num_pipeline = Pipeline([('selector', DataFrameSelector(list_num_attribs)), ("imputer", Imputer(strategy='median')), (
            "attribs_adder", CombinedAttributesAdder()), ('std_scaler', StandardScaler())])

        #num_pipeline_tr = num_pipeline.fit_transform(housing_num)
        return num_pipeline

    def create_categorical_pipeline(self):
        cat_attribs = self.cat_attribs
        cat_pipeline = Pipeline([('selector', DataFrameSelector(
            cat_attribs)), ('cat_encoder', CategoricalEncoder(encoding='onehot-dense'))])
        return cat_pipeline

    def full_pipeline(self):
        self.num_pipeline = self.create_numerical_pipeline()
        self.cat_pipeline = self.create_categorical_pipeline()

        self.final_pipeline = FeatureUnion(transformer_list=[(
            'num_pipeline', self.num_pipeline), ('cat_pipeline', self.cat_pipeline), ])
        return self.final_pipeline

    def final_transformed_data(self):
        print("Housing ==\n", self.housing.head())

        self.final_pipeline = self.full_pipeline()

        self.final_prepared_data = self.final_pipeline.fit_transform(
            self.housing)

        print("Type Final Prepared Data \n\n", type(
            self.final_prepared_data), self.final_prepared_data)

        print("Final Prepared Data Row[1]=\n",
              list(self.final_prepared_data[0]))

        print("Final Label\n\n", type(self.housing_labels))

        cats_encoder = self.cat_pipeline.named_steps['cat_encoder']
        print("Categories = ", cats_encoder.categories_)
        return self.final_prepared_data

    def get_housing_data_components(self):
        return self.housing, self.housing_num, self.housing_labels

    def get_pipeline_components(self):
        return self.final_pipeline, self.num_pipeline, self.cat_pipeline


if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'

    analysis = Analysis(HousingData()).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()

    data_preparation = DataPreparation(strat_train_set, strat_test_set)
    #final_prepared_data, housing_labels, full_pipeline = data_preparation.final_transformed_data(strat_train_set, strat_test_set)
    final_prepared_data = data_preparation.final_transformed_data()
