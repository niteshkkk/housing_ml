import matplotlib.pyplot as plt
import mpld3
import os
import config as conf


def save_as_html(file_name):
    if not file_name:
        return
    path_to_save = os.path.join(conf.MPLD3_HTML_FILES_DIR, file_name)
    mpld3.save_html(plt.gcf(), path_to_save + ".html")


def save_figure_as_html(file_name):
    if not file_name:
        return
    path_to_save = os.path.join(conf.MPLD3_HTML_FILES_DIR, file_name)
    print("\n\n gcf = \n\n", plt.gcf())
    mpld3.save_html(plt.gcf(), path_to_save + ".html")
