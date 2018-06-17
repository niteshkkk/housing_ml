import os
import tarfile
from six.moves import urllib
import requests

download_root = "https://github.com/ageron/handson-ml/raw/master/"
path = "datasets/housing"
url = download_root + path + "/housing.tgz"


def fetch_housing_data(housing_url=url, housing_path=path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def url_retrieve(url, path, name, force=False):
    if os.path.isfile(os.path.join(path, name)) and not force:
        print("Found File")
        return os.path.abspath(os.path.join(path, name))
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, name)
    urllib.request.urlretrieve(url, file_path)
    return os.path.abspath(file_path)


def fetch_housing_data_this_also_works_well(housing_url=url, housing_path=path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    with open(tgz_path, "wb") as file:
        response = requests.get(housing_url)
        file.write(response.content)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


if __name__ == "__main__":
    # fetch_housing_data()
    url_retrieve('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json',
                 'datasets/newsgroup', 'a.json')
