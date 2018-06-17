import os
import sys
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import mpld3_graphs.utils as utils
from ml.base_data.base_data_frame import BaseDataFrame

download_root = "https://github.com/ageron/handson-ml/tree/master/"
path = "datasets/housing"
url = download_root + path + "/housing.tgz"
mpl3_html_path = "/Users/niteshk/svn_tree/machine-learning/scikit/housing-project/mpld3_html"


class HousingData(BaseDataFrame):

    def __init__(self, housing_path=path):
        self.load_housing_data(path)
        super().__init__(self.housing_data_frame)

    def load_housing_data(self, housing_path=path):
        csv_path = os.path.join(housing_path, "housing.csv")
        self.housing_data_frame = pd.read_csv(csv_path)

    def plot_and_save_as_html(self):
        if self.housing_data_frame is None:
            print("No Housing Data to Plot. Refresh data using load_housing_data() .... ")
            return

        self.housing_data_frame.hist(bins=50, figsize=(10, 7))

        # This works: mpld3.show()

        utils.save_as_html('housing_histogram')

    def mpld3_show(self):
        self.housing_data_frame.hist(bins=50, figsize=(10, 7))
        mpld3.show()

    def save_as_html_deprecated(self):
        saved_path = os.path.join(mpl3_html_path, "housing_histogram")
        mpld3.save_html(plt.gcf(), saved_path + ".html")


if __name__ == "__main__":
    housing = HousingData()
    housing.plot_and_save_as_html()
    print("\n\n==== Head ====\n", housing.head())
    print("\n\n==== Info ====\n", housing.info())
    print("\n\n==== Describe ====\n", housing.describe())

    print(len(sys.argv))

    if len(sys.argv) > 1 and sys.argv[1] == 'show':
        print("\n\n\nRunning Mpld3 Graphs in Browser.....")
        housing.mpld3_show()
