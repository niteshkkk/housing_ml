import mpld3_graphs.utils as utils
import matplotlib.pyplot as plt


class BaseAnalysis(object):

    def __init__(self, to_show=False):
        self.to_show = to_show

    def show(self):
        if self.to_show:
            plt.show()

    def plot_and_save_hist(self, data_frame, column, to_save_file_name):

        data_frame[column].hist()
        # self.show()
        utils.save_as_html(to_save_file_name)

    def plot_and_save_scatter(self, data_frame, x, y, alpha, to_save_file_name):
        # print(x, y, alpha, to_save_file_name)
        # kwargs = dict(kind='scatter', x=x, y=y, alpha=alpha)
        # data_frame.plot(kind='scatter', x=x1, y=y1, alpha=alpha1)
        data_frame.plot(kind='scatter', x=x, y=y, alpha=alpha)
        # self.show()
        utils.save_figure_as_html(to_save_file_name)

    def plot_and_save_scatter_details(self, data_frame, x, y, alpha, to_save_file_name, **kwds):
        # print(x, y, alpha, to_save_file_name)
        # kwargs = dict(kind='scatter', x=x, y=y, alpha=alpha)
        # data_frame.plot(kind='scatter', x=x1, y=y1, alpha=alpha1)
        data_frame.plot(kind='scatter', x=x, y=y, alpha=alpha, **kwds)
        # self.show()
        utils.save_figure_as_html(to_save_file_name)
