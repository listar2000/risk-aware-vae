import scipy.io as sio
import numpy as np


class DataSet(object):
    """
    Class to manage your data
    If you need to use batch for your algorithm, there is a next_batch implementation
    """

    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


"""
#    Next batch operation
    def next_batch(self, batch_size):

        # Batch size should be smaller than sample size
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
"""


def read_mnist(datapath):
    """
    @datapath : path to the input data
    Read data
    """

    data = sio.loadmat(datapath)
    train = DataSet(data['train'], data['trainLabel'])
    tune = DataSet(data['tune'], data['tuneLabel'])
    test = data['test']

    return train, tune, test


class DataSet_Twoview(object):
    """
    Class to manage your data
    If you need to use batch for your algorithm, there is a next_batch implementation
    """

    def __init__(self, images, images2, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._images2 = images2
        self._labels = labels
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


def read_mnist_twoview(datapath):
    """
    @datapath : path to the input data
    Read data
    """

    data = sio.loadmat(datapath)
    train = DataSet_Twoview(data['train'], data['train2'], data['trainLabel'])
    tune = DataSet_Twoview(data['tune'], data['tune2'], data['tuneLabel'])
    test1 = data['test']
    test2 = data['test2']

    return train, tune, test1, test2


"""
#    Next batch operation
    def next_batch(self, batch_size):

        # Batch size should be smaller than sample size
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

"""