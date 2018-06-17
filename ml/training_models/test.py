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
cat_attribs = ["ocean_proximity"]


def numerical_analysis(strat_train_set, strat_test_set):
    print("\n\n --------- Cleaning ----------- \n\n")

    strat_train_set.hist(bins=50, figsize=(8, 8))
    housing = strat_train_set

    strat_train_set_num = strat_train_set.drop("ocean_proximity", axis=1)
    print("List housing num = ", list(strat_train_set_num),
          list(strat_train_set_num.columns.values))
    imputer = Imputer(strategy="median")

    X = imputer.fit_transform(strat_train_set_num)

    # strat_train_set_imputed = pd.DataFrame(
    # X, columns=strat_train_set_num.columns, index=strat_train_set_num.index)

    X1 = StandardScaler().fit_transform(X)

    # scaled_train_set = StandardScaler().fit_transform(
    # strat_train_set_imputed)

    scaled_train_set = pd.DataFrame(
        X1, columns=strat_train_set_num.columns, index=strat_train_set_num.index)

    scaled_train_set.hist(bins=50, figsize=(8, 8))
    print(scaled_train_set.head(20))
    return strat_train_set_num


def categorical_analysis(strat_train_set, strat_test_set):
    data = np.array(['cold', 'cold', 'warm', 'cold', 'hot',
                     'hot', 'warm', 'cold', 'warm', 'hot'])

    print(type(data), data)


def hot_encoding(strat_train_set, strat_test_set):
    print("---- \n\n In Hot Encoding -------- \n\n", type(housing))
    housing_cat = housing[['ocean_proximity']]
    print("Type = ", type(housing_cat))
    print(housing_cat.head(10))
    print("Type = ", type(housing_cat.values), housing_cat.values)
    housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
    print("Housing cat reshaped = ", housing_cat_reshaped)
    housing_cat_1hot = CategoricalEncoder().fit_transform(housing_cat_reshaped)
    print("Housing cat hot encoded = ", type(
        housing_cat_1hot), housing_cat_1hot.toarray())


def add_custom_attr(strat_train_set, strat_test_set):
    print("\n\n Add Cutom Attributes \n\n")
    housing = strat_train_set
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    print(type(housing_extra_attribs), housing_extra_attribs)

    a = np.array([[2, 3, 5], [6, 79, 9]])
    b = a[:, 1]
    print(type(b), b)
    print(type(a), a)
    c = np.c_[a, b]
    print(type(c), c)

    print(type(housing.columns), housing.columns)

    housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(
        housing.columns) + ["rooms_per_househole", "population_per_household"])
    print(housing_extra_attribs.head(10))


def create_numerical_pipeline(strat_train_set, strat_test_set):
    housing_num = strat_train_set.drop("ocean_proximity", axis=1)
    num_pipeline = Pipeline([("imputer", Imputer(strategy='median')), (
        "attribs_adder", CombinedAttributesAdder()), ('std_scaler', StandardScaler())])

    #num_pipeline_tr = num_pipeline.fit_transform(housing_num)
    print(type(num_pipeline_tr))
    return num_pipeline


def create_categorical_pipeline(strat_train_set, strat_test_set):
    cat_pipeline = Pipeline([('seelctor', DataFrameSelector(
        cat_attribs)), ('cat_encoder', CategoricalEncoder(encoding='onehot-dense'))])
    return cat_pipeline


if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'

    analysis = Analysis(HousingData()).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()
    numerical_analysis(strat_train_set, strat_test_set)
    housing = strat_train_set

    #### Hot Encoding ######

    categorical_analysis(strat_train_set, strat_test_set)
    hot_encoding(strat_train_set, strat_test_set)
    add_custom_attr(strat_train_set, strat_test_set)
    create_numerical_pipeline(strat_train_set, strat_test_set)

    print("\n\n\n------- Data Frame Selector ---------------- \n\n\n")
    X = DataFrameSelector(cat_attribs).fit_transform(housing)
    print(type(X), X)

    if show:
        plt.show()
