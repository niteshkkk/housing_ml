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
from ml.graph_analysis.hist_graph import Analysis
from pandas.plotting import scatter_matrix

np.random.seed(42)
test_ratio = 0.2
random_state = 42


class AnalysisPlots(BaseAnalysis):

    def __init__(self, strat_train_set, strat_test_set, show):
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set
        self.housing = self.strat_train_set.copy()
        super().__init__(show)

    def scatter_plot(self):
        housing = self.housing
        super().plot_and_save_scatter(housing, 'longitude', 'latitude',
                                      0.1, "lat_long_scatter")
        #housing.plot(kind="scatter", x='longitude', y='latitude', alpha=0.1)
        # plt.show()
        # utils.save_as_html("lat_long_scatteer1")

        kwds = dict(s=housing["population"] / 100, label="population", figsize=(10, 7),
                    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                    sharex=False)
        # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        #             s=housing["population"] / 100, label="population", figsize=(10, 7),
        #             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        #             sharex=False)
        # plt.legend()
        super().plot_and_save_scatter_details(
            housing, 'longitude', 'latitude', 0.4, "lat_long_population_price", **kwds)
        # plt.show()

    def corr(self):
        housing = self.housing
        corr_matrix = housing.corr()
        corr_median_house_value = corr_matrix[
            'median_house_value'].sort_values(ascending=False)
        print("\n\n Correlation Matrix \n\n", corr_matrix)
        print("\n\n Correlation Matrix for Median House Value\n\n",
              corr_median_house_value)

    def imp_corr(self):
        attrs = ['median_house_value', 'median_income',
                 'total_rooms', 'housing_median_age']

        scatter_matrix(self.housing[attrs], figsize=(8, 6))

        super().plot_and_save_scatter(self.housing, 'median_income',
                                      'median_house_value', 0.1, 'median_house_value_vs_median_income')

    def pipeline(self):
        pass

if __name__ == "__main__":
    show = len(sys.argv) > 1 and sys.argv[1] == 'show'
    analysis = Analysis(HousingData()).pipeline()
    strat_train_set, strat_test_set = analysis.get_stratified_train_test_set()
    analysis_plot = AnalysisPlots(strat_train_set, strat_test_set, show)
    analysis_plot.scatter_plot()
    analysis_plot.corr()
    analysis_plot.imp_corr()

    if show:
        plt.show()
