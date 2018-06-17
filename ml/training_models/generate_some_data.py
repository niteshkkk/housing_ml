import numpy as np
import matplotlib.pyplot as plt
import mpld3
import mpld3_graphs.utils as utils
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class KVExtractor(TransformerMixin):

    def __init__(self, kvpairs):
        self.kpairs = kvpairs

    def transform(self, X, *_):
        print("In transform 1")
        result = []

        return result

    def fit(self, *_):
        print("In Fit 1")
        return self

X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# print(X)
# print(Y)


plt.scatter(X, Y)

a = np.random.rand(3, 4)
# print(a)
# plt.show()


D = pd.DataFrame([['a', 1, 'b', 2], ['b', 2, 'c', 3]],
                 columns=['k1', 'v1', 'k2', 'v2'])
kvpairs = [['k1', 'v1'], ['k2', 'v2']]
r = KVExtractor(kvpairs).fit_transform(D)
print(r)

pipeline = Pipeline(steps=[
    ('s1', KVExtractor(kvpairs)),
    ('s2', KVExtractor(kvpairs))
])

print("\n\n ---- In Pipeline Fit\n\n")
pipeline.fit(D)

print("\n\n In Pipeline Transform\n\n")
pipeline.transform(D)

print("\n\n In Pipeline fit-Transform\n\n")
pipeline.fit_transform(D)

a = np.array([1, 2.3])
print("Nd Array E.g: \n\n Mean: , Std Deviation: ", type(
    a), a.mean(), a.std())
