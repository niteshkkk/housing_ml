import matplotlib.pyplot as plt
import mpld3

fig = plt.figure()
plot = plt.plot([3, 1, 4, 1, 5], 'ks-', mec='w', mew=5, ms=20)
print(plot)
# print(type(fig))
# plt.show(fig)
mpld3.save_html(
    fig, "/Users/niteshk/svn_tree/machine-learning/scikit/housing-project/mpld3-html/a.html")
