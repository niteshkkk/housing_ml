import numpy as np
from zlib import crc32
import hashlib
from sklearn.model_selection import train_test_split
from ml.download_data.load_housing_data import HousingData

np.random.seed(42)
test_ratio = 0.2
random_state = 42


class GenerateTrainingSet(object):

    def __init__(self, data):
        self.data = data

    def split_train_set(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data)) * test_ratio
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def print_test_train_set(self, data):
        train_set, test_set = self.split_train_set(data, test_ratio)
        print("Train ", len(train_set))
        print("\nTest ", len(test_set))

    def test_set_check(self, identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    def split_train_test_by_id(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(
            lambda id_: self.test_set_check(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    def test_set_check_hash(self, identifier, test_ratio, hash=hashlib.md5):
        return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def test_set_check_hash_byte_array(self, identifier, test_ratio, hash=hashlib.md5):
        return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

    def add_index_as_id_and_split_training_set(self):
        print("\n\n add_index_as_id_and_split_training_set \n\n")
        data_with_id = self.data.reset_index()   # adds an `index` column

        train_set, test_set = self.split_train_test_by_id(
            data_with_id, 0.2, "index")

        print(test_set.head())
        return train_set, test_set

    def add_custom_id_as_index_and_split_training_set_deprecated(self):
        print("\n\n add_custom_id_as_index_and_split_training_set_0 \n\n")
        data_with_id = self.data
        data_with_id["id"] = self.data[
            "longitude"] * 1000 + data["latitude"]

        train_set, test_set = self.split_train_test_by_id(
            data_with_id, 0.2, "id")

        print(test_set.head())
        return train_set, test_set

    def add_custom_id_as_function_and_split_training_set(self, func):
        print("\n\n add_custom_id_as_function_and_split_training_set \n\n")
        data_with_id = self.data
        data_with_id["id"] = func(self.data)
        train_set, test_set = self.split_train_test_by_id(
            data_with_id, 0.2, "id")
        print(test_set.head())
        return train_set, test_set

    def split_train_set_sklearn(self, test_size, random_state):
        print("\n\n se sklearn and split train set \n\n")
        train_set, test_set = train_test_split(
            self.data, test_size=test_size, random_state=random_state)

        print(test_set.head())
        return train_set, test_set

if __name__ == "__main__":
    housing = HousingData().get_data_frame()
    training_set = GenerateTrainingSet(housing)
    training_set.add_index_as_id_and_split_training_set()
    training_set.add_custom_id_as_function_and_split_training_set(
        lambda data: data['longitude'] * 1000 + data['longitude'])
    training_set.split_train_set_sklearn(test_ratio, random_state)
