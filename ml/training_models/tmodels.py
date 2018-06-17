from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import numpy as np
import pandas as pd
import sys
from zlib import crc32
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder
from ml.download_data.load_housing_data import HousingData
from ml.test_set.split_test_set import GenerateTrainingSet
import mpld3_graphs.utils as utils
from ml.graph_analysis.analysis_plots import Analysis
from encoder import CategoricalEncoder
from ml.transformer import CombinedAttributesAdder, DataFrameSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from ml.training_models.data_preparation import DataPreparation
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TModels(object):

    def __init__(self, data_preparation):
        self.data_preparation = data_preparation
        self.final_prepared_data = self.data_preparation.final_transformed_data()
        self.housing, self.housing_num, self.housing_labels = self.data_preparation.get_housing_data_components()
        self.full_pipeline, self.num_pipeline, self.cat_pipeline = self.data_preparation.get_pipeline_components()
        self.prediction_df = None
        self.lin_scores = None
        self.tree_scores = None
        self.random_forest_score = None
        self.svr_score = None

    def linear_regression(self):
        final_prepared_data = self.final_prepared_data
        housing_labels = self.housing_labels
        housing = self.housing
        housing_num = self.housing_num
        full_pipeline = self.full_pipeline

        lin_reg = LinearRegression()
        lin_reg.fit(final_prepared_data, housing_labels)

        some_data = housing.iloc[:5]
        some_labels = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)

        print("Predictions:", lin_reg.predict(some_data_prepared))
        print("Label:", list(some_labels))
        print("Label as Nd array:", list(some_labels.values))

        housing_predictions = lin_reg.predict(final_prepared_data)

        print("Type predictions: ", type(housing_predictions),
              "\n Type Labels: ", type(housing_labels), "\n Type Label.values: ", type(some_labels.values))

        print("\n\n------------------  Linear Regression RMSE and MAE \n")
        self.calculate_error(self.housing_labels, housing_predictions)

        lin_scores = cross_val_score(
            lin_reg, final_prepared_data, housing_labels, scoring="neg_mean_squared_error", cv=10)
        self.lin_scores = lin_scores

        self.prediction_df = self.get_prediction_df()
        print(type(housing_predictions))
        self.prediction_df['linear_predictions'] = housing_predictions
        self.prediction_df['linear_absolute_error'] = np.absolute(
            self.prediction_df['median_house_value'] - self.prediction_df['linear_predictions'])
        self.prediction_df['linear_percentage_error'] = (self.prediction_df[
            'linear_absolute_error'] / self.prediction_df['median_house_value']) * 100

    def dec_tree_regression(self):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.final_prepared_data, self.housing_labels)
        housing_predictions = tree_reg.predict(self.final_prepared_data)

        print("\n\n----------------  Decision Tree RMSE and MAE \n")
        self.calculate_error(self.housing_labels, housing_predictions)

        scores = cross_val_score(tree_reg, self.final_prepared_data,
                                 self.housing_labels, scoring="neg_mean_squared_error", cv=10)
        self.tree_scores = scores

        self.prediction_df = self.get_prediction_df()
        self.prediction_df['tree_predictions'] = housing_predictions
        self.prediction_df['tree_absolute_error'] = np.absolute(
            self.prediction_df['median_house_value'] - self.prediction_df['tree_predictions'])

    def calculate_error(self, labels, predictions):
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)

        print("\nRMSE ( Root mean squared error): \n",
              rmse, "\n\nMean Absolute Error:\n", mae)

    def random_forest_regression(self):
        forest_reg = RandomForestRegressor(random_state=42)
        forest_reg.fit(self.final_prepared_data, self.housing_labels)
        housing_predictions = forest_reg.predict(self.final_prepared_data)

        print("\n\n------------  Random Forest Regression RMSE and MAE \n")
        self.calculate_error(self.housing_labels, housing_predictions)

        self.random_forest_score = cross_val_score(forest_reg, self.final_prepared_data,
                                                   self.housing_labels, scoring="neg_mean_squared_error", cv=10)
        self.prediction_df = self.get_prediction_df()
        self.prediction_df['forest_predictions'] = housing_predictions
        self.prediction_df['forest_absolute_error'] = np.absolute(
            self.prediction_df['median_house_value'] - self.prediction_df['forest_predictions'])

    def svr(self):
        svm_reg = SVR(kernel='linear')
        svm_reg.fit(self.final_prepared_data, self.housing_labels)
        housing_predictions = svm_reg.predict(self.final_prepared_data)
        print("\n\n---------------- SVM RMSE and MAE errors\n")
        self.calculate_error(self.housing_labels, housing_predictions)

        self.svr_score = cross_val_score(
            svm_reg, self.final_prepared_data, self.housing_labels, scoring='neg_mean_squared_error', cv=10)
        self.prediction_df = self.get_prediction_df()
        self.prediction_df['svr_predictions'] = housing_predictions
        self.prediction_df['svr_absolute_error'] = np.absolute(
            self.prediction_df['median_house_value'] - self.prediction_df['svr_predictions'])

    def grid_search(self):
        param_grid = [{'n_estimators': [3, 10, 30],
                       'max_features': [2, 4, 6, 8]},
                      {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(self.final_prepared_data, self.housing_labels)
        print("\n\n--------------- Grid Search\n ",
              grid_search, "\n\n Grid Search Test scores:\n")
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        print("\n\n Grid search best parameter: ", grid_search.best_params_)
        print("\n\n Gird search best estimators: ", grid_search.best_estimator_)
        grid_df = pd.DataFrame(grid_search.cv_results_)
        print("\n\n Grid Search Data Frame\n", grid_df)

    def display_scores(self):
        lin_rmse_scores = np.sqrt(-self.lin_scores)
        print("\nCross Validation scores for Linear Regression")
        self.__display_scores_details(lin_rmse_scores)

        tree_rmse_scores = np.sqrt(-self.tree_scores)
        print("\nCross Validation score for Tree Regression")
        self.__display_scores_details(tree_rmse_scores)

        forest_scores = np.sqrt(-self.random_forest_score)
        print("\nCross Validation score fro Random Forest Regression")
        self.__display_scores_details(forest_scores)

        svr_scores = np.sqrt(-self.svr_score)
        print("\nCross Validation score for State Vector Regression")
        self.__display_scores_details(svr_scores)

    def display_post_prediction_matrix(self):
        print("\n\n-------------- Prediction Data Frame \n\n",
              self.get_prediction_df())

    def __display_scores_details(self, scores):
        print("\n\nCross Validataion \n\n")
        print("Scores: ", scores)
        print("Mean: ", scores.mean())
        print("Standard Deviation: ", scores.std())

    def get_prediction_df(self):
        if self.prediction_df is not None:
            return self.prediction_df
        self.prediction_df = self.housing_labels.copy().to_frame()
        return self.prediction_df


class CrossValidation(object):

    def __init__(self):
        pass

if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'

    analysis = Analysis(HousingData()).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()

    data_preparation = DataPreparation(strat_train_set, strat_test_set)
    tmodels = TModels(data_preparation)
    tmodels.linear_regression()
    tmodels.dec_tree_regression()
    tmodels.random_forest_regression()
    tmodels.svr()
    tmodels.display_post_prediction_matrix()
    tmodels.display_scores()
    tmodels.grid_search()
    print(zip(1, 2))
