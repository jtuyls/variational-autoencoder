# Code for downloading and loading MNIST
# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb
import os
import gzip
import cPickle
import numpy as np

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

    train_x, train_y = train_set
    train_x = train_x.reshape(train_x.shape[0], 1, 28, 28)
    print('... done loading data')
    return train_x #/ np.float32(255)
