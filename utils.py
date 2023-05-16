import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


def visualize_dataset_in_grid(images: torch.Tensor, labels: torch.Tensor = None,
                              num_rows=3, num_cols=3, fig_size=(8, 8)):
    num_images = num_rows * num_cols
    assert len(images) >= num_images, f"not enough images for {num_rows} rows and {num_cols} cols"
    assert len(fig_size) == 2, "invalid fig_size"

    # Select the first 25 images and labels from the batch
    images = images[:num_images]
    show_label = labels is not None
    if show_label:
        labels = labels[:num_images]

    # Create a plot with 5 x 5 subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Loop over each image and label in the batch and plot it in a subplot
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].imshow(image.squeeze(), cmap="gray")
        if show_label:
            axs[row, col].set_title(f"Label: {labels[i]}")
        axs[row, col].axis("off")

    # Show the plot
    plt.show()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="gray")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def grid_show(imgs):
    """
    Display images (B x C x W x H) on a grid
    """

    img_grid = make_grid(imgs)
    matplotlib_imshow(img_grid, one_channel=True)


def generate_img(model, z_dim):
    """
    Generate a random sample of images from a VAE model
    """

    with torch.no_grad():
        z = torch.randn(model.batch_size, z_dim).cuda()
        sample = model.model.decode(z)
    return sample


def reconstruct_img(model, x):
    """
    Reconstruct a image by doing one forward pass of a VAE model
    """ 

    with torch.no_grad():
        sample, _, _ = model.model(x.cuda())
    return sample


def compute_IS(sample):
    """
    Compute the Inception Score for a batch of generated samples
    """

    # Only compatible with MNIST for now
    inception = InceptionScore(normalize=True)
    inception.update(sample.repeat(1, 3, 1, 1))
    return inception.compute()


def compute_FID(train, sample):
    """
    Compute the Frechet Inception Distance for a batch of reconstructed samples
    """

    # Only compatible with MNIST for now
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    img_dist1 = train.repeat(1, 3, 1, 1)
    img_dist2 = sample.repeat(1, 3, 1, 1)
    fid.update(img_dist1, real=True)
    fid.update(img_dist2, real=False)
    return fid.compute()