import numpy as np
import torch
import torch.utils.data as tud
from torchvision import datasets
from torchvision.transforms import ToTensor
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


def read_mnist(datapath="./data"):
    """
    Read MNIST data
    """

    training_data = datasets.MNIST(
        root=datapath,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=datapath,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data


def read_fashionmnist(datapath="./data"):
    """
    Read Fashion-MNIST data
    """

    training_data = datasets.FashionMNIST(
        root=datapath,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=datapath,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data


def read_cifar10(datapath="./data"):
    """
    Read CIFAR10 data
    """

    training_data = datasets.CIFAR10(
        root=datapath,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root=datapath,
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data
