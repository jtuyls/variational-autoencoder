# Code for downloading and loading MNIST
# https://github.com/mllfreiburg/dl_lab_2016/blob/master/notebooks/exercise_1.ipynb
import os
import gzip
import cPickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_paths import data_path, data_path_cells

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
    test_x, test_y = test_set
    test_x = test_x.reshape(test_x.shape[0], 1, 28, 28)
    print('... done loading data')
    return train_x, val_x, test_x, test_y #/ np.float32(255)

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
    # Return the male column of the labels since we want to see how good a vae can unsupervisedly make a distinction between males and females
    index = attr_names.index("Male")
    test_labels = test_labels[:, index]
    print(test_labels.shape)
    print("... done loading data")
    return train_images, val_images, test_images, test_labels

def cell_data(name="Brightfield_Pollen"):
    print("Loading cell data")
    pollen_image_name = name + "_cell{}_masked.tif"
    image_path = data_path_cells + pollen_image_name if data_path_cells[-1] == "/" else data_path_cells + "/" + pollen_image_name

    images = []
    another_image = True
    i = 0
    while another_image:
        try:
            complete_number_string = complete_string(str(i))
            image = plt.imread(image_path.format(complete_number_string))
            images.append(image)
            i += 1
        except IOError:
            another_image = False
    images = np.float32(np.array(images)) / 255.0
    images = images.reshape(images.shape[0], 1, 64, 64)
    folds = np.array_split(images, 10)
    X_train = np.concatenate(folds[0:8])
    X_val = folds[8]
    X_test = folds[9]
    print("... done loading data")
    return X_train, X_val, X_test, np.array([])

def cell_data_input_output(name="Brightfield_Pollen"):
    print("Loading cell data")
    pollen_image_input_name = name + "_cell{}.tif"
    pollen_image_output_name = name + "_cell{}_masked.tif"
    image_input_path = data_path_cells + pollen_image_input_name if data_path_cells[
                                                            -1] == "/" else data_path_cells + "/" + pollen_image_input_name
    image_output_path = data_path_cells + pollen_image_output_name if data_path_cells[
                                                                        -1] == "/" else data_path_cells + "/" + pollen_image_output_name

    input_images = []
    output_images = []
    another_image = True
    i = 0
    while another_image:
        try:
            complete_number_string = complete_string(str(i))
            input_image = plt.imread(image_input_path.format(complete_number_string))
            input_images.append(input_image)
            output_image = plt.imread(image_output_path.format(complete_number_string))
            output_images.append(output_image)
            i += 1
        except IOError:
            another_image = False
    input_images = np.float32(np.array(input_images)) / 255.0
    input_images = input_images.reshape(input_images.shape[0], 1, 64, 64)
    folds = np.array_split(input_images, 10)
    X_train = np.concatenate(folds[0:8])
    X_val = folds[8]
    X_test = folds[9]

    output_images = np.float32(np.array(output_images)) / 255.0
    output_images = input_images.reshape(output_images.shape[0], 1, 64, 64)
    folds = np.array_split(output_images, 10)
    Y_train = np.concatenate(folds[0:8])
    Y_val = folds[8]
    Y_test = folds[9]

    print("... done loading data")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def complete_string(number_string):
    while len(number_string) < 4:
        number_string = "0" + number_string
    return number_string



