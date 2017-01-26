# Code for downloading and loading MNIST
# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb
import os
import gzip
import cPickle
import numpy as np

from data_paths import data_path

def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_x, _ = train_set
    train_x = train_x.reshape(train_x.shape[0], 1, 28, 28)
    val_x, _ = valid_set
    val_x = val_x.reshape(val_x.shape[0], 1, 28, 28)
    print('... done loading data')
    return train_x, val_x #/ np.float32(255)

def celeb_data():
        print("Loading celeb data")
        train_images = np.float32(np.load(os.path.join(
                data_path, "train_images_32.npy"))) / 255.0
        train_labels = np.uint8(np.load(os.path.join(
                data_path, "train_labels_32.npy")))
        val_images = np.float32(np.load(os.path.join(
                data_path, "val_images_32.npy"))) / 255.0
        val_labels = np.uint8(np.load(os.path.join(
                data_path, "val_labels_32.npy")))
        test_images = np.float32(np.load(os.path.join(
                data_path, "test_images_32.npy"))) / 255.0
        test_labels = np.uint8(np.load(os.path.join(
                data_path, "test_labels_32.npy")))

        with open(os.path.join(data_path, "attr_names.txt")) as f:
            attr_names = f.readlines()[0].split()

        return train_images, val_images
