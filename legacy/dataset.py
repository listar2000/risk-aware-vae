import scipy.io as sio


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