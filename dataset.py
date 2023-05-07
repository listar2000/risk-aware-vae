import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as tud
from torch.distributions.multivariate_normal import MultivariateNormal
import os


class FakeDataset(tud.Dataset):
    """
    A fake dataset for demo usage
    """
    def __init__(self, dim=100, num_class=10, num_sample=10000, file_path=None, save_file=False):
        assert not num_sample % num_class, "num_sample must be multiples of num_class"
        self.num_class = num_class
        self.num_sample = num_sample
        self.dim = dim

        self.ys, self.xs = self.create_fake_data(file_path, save_file)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx]

    def create_fake_data(self, file_path=None, save_file=False):
        if file_path and os.path.isfile(file_path):
            data = np.load('data.npz')
            xss_np = data['xss']
            ys_np = data['ys']
            return torch.from_numpy(ys_np), torch.from_numpy(xss_np)

        rep = self.num_sample // self.num_class
        ys = torch.repeat_interleave(torch.arange(10), repeats=rep)
        cor_xs = torch.arange(10) * 0.1
        xss = []
        for i in range(10):
            cov_mat = torch.Tensor([[1, cor_xs[i]], [cor_xs[i], 1]])
            dist = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=cov_mat)
            samps = dist.sample(sample_shape=(rep,))
            coords = torch.randint(0, 100, size=(2,))
            xs = torch.zeros((rep, self.dim))
            xs[:, coords] = samps
            xss.append(xs)

        xss = torch.vstack(xss)
        assert len(ys) == len(xss)

        if save_file:
            file_path = file_path or "./data/fake.npz"
            xss_np = xss.numpy()
            ys_np = ys.numpy()
            # Save the arrays to a .npz file
            np.savez(file_path, xss=xss_np, ys=ys_np)
            print(f"saving fake data into {file_path}")

        return ys, xss


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