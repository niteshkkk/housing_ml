import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def startified_sampling_test(self):
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    print(X)
    Y = np.array([0, 0, 1, 1])
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for train_i, test_i in s.split(X, Y):
        print("train i = ", train_i, " test i = ", test_i)
        print(X[train_i])
        print(X[test_i])


d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df['one'])
print(df['two']['a'])

df_id = df
df_id['id'] = df['one'] * 10 + df['two']
print(df_id)

print(df_id.loc['b'])


print("----- Next Example - --------------------------")

df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])

df = df.append(df2)
print(df)
print(df.iloc[1])

print("------- NP permutations -------------")
shuffled_indices = np.random.permutation(10)
print(shuffled_indices)
test_indices = shuffled_indices[:7]
print(id(shuffled_indices))
print(id(test_indices))
