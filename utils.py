"""
Code for visualization, metric computation and model evaluation.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from scipy import stats


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


def show_gen_img(model, z_dim):
    sample = generate_img(model, z_dim)
    sample = sample.view(model.batch_size, 1, 28, 28).cpu()
    grid_show(sample)


def show_recon_img(model, x):
    sample = reconstruct_img(model, x)
    imgs = torch.cat((x.view(1, 1, 28, 28).cpu(), sample.view(1, 1, 28, 28).cpu()))
    grid_show(imgs)


def knn(Mxx, Mxy, Myy, k=1, sqrt=True):
    """
    The leave-one-out accuracy of a 1-NN classifier.
    Input: L2 distances in some feature space (important: not pixel space)
    credit: https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
    """
    
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy),1), torch.cat((Mxy.transpose(0,1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    _, idx = (M + torch.diag(INFINITY * torch.ones(n0+n1))).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = torch.eq(label, pred).float().mean()

    return acc, precision, recall


def mmd(Mxx, Mxy, Myy, sigma = 1):
    """
    Kernel Maximum Mean Discrepancy (Gaussian kernel).
    Input: L2 distances in some feature space (important: not pixel space)
    credit: https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
    """

    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma ** 2))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma ** 2))
    Myy = torch.exp(-Myy / (scale * 2 * sigma ** 2))
    a = Mxx.mean() + Myy.mean() - 2 * Mxy.mean()
    mmd = max(a, 0) ** (0.5)

    return mmd


def IS(sample):
    """
    WIP.
    Compute the Inception Score for a batch of generated samples
    Note: Only use with Imagenet, need to investigate minimum sample size
    """

    # Only compatible with MNIST for now
    inception = InceptionScore(normalize=True)
    inception.update(sample.repeat(1, 3, 1, 1))
    return inception.compute()


def FID(train, sample):
    """
    WIP.
    Compute the Frechet Inception Distance for a batch of reconstructed samples
    Note: Only use with Imagenet, need to investigate minimum sample size
    """

    # Only compatible with MNIST for now
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    img_dist1 = train.repeat(1, 3, 1, 1)
    img_dist2 = sample.repeat(1, 3, 1, 1)
    fid.update(img_dist1, real=True)
    fid.update(img_dist2, real=False)
    return fid.compute()


def compute_recon_loss(model, val_dataloader):
    """
    Given a validation set, get all reconstructed samples and associated losses
    """
    
    list_recon_loss = []
    list_recon_samples = []
    for val_features, _ in tqdm(val_dataloader):
        b = val_features.shape[0]
        val_samples = val_features.view(b, -1).cuda()
        recon_samples = reconstruct_img(model, val_samples)
        list_recon_samples.append((val_samples, recon_samples))
        # loss dimension: batch size x (W x H)
        recon_loss = model.recon_loss_f(recon_samples, val_samples, reduction="none")
        list_recon_loss.append(recon_loss)

    return (list_recon_samples, list_recon_loss)


def reorder(list_recon_loss, by):
    """
    Reshape the list of reconstruction loss
    """
    
    if by == "image":
        return [img_loss.cpu() for loss in list_recon_loss for img_loss in loss.sum(1)]
    elif by == "image_nested":
        return [loss.sum(1).cpu() for loss in list_recon_loss]
    elif by == "batch":
        return [loss.sum(1).mean().cpu() for loss in list_recon_loss]


def plot_recon_loss(recon_loss_a, recon_loss_b):
    """
    Plot reconstruction loss as histograms
    """
    
    plt.hist(recon_loss_a, density=True)
    plt.hist(recon_loss_b, density=True)
    plt.show()
            

def best_batch(samples, loss_by_batch, loss_by_image_nested):
    print("Loss on best batch:", np.min(loss_by_batch))
    best_idx = np.argmin(loss_by_batch)
    return (samples[best_idx], loss_by_image_nested[best_idx])


def plot_best_batch(best_batch):
    best_batch_in, best_batch_out = best_batch
    best_batch_in = best_batch_in.view(64, 1, 28, 28).cpu()
    best_batch_out = best_batch_out.view(64, 1, 28, 28).cpu()
    grid_show(torch.cat((best_batch_in, best_batch_out), 3))
    plt.show()


def batch_t_test(recon_loss_a, recon_loss_b):
    return stats.ttest_ind(recon_loss_a, recon_loss_b)


def plot_best_images(samples, loss, n=3):
    best_images_idx = np.argsort(loss)[:n]
    for idx in best_images_idx:
        print("Reconstruction loss:", loss[idx].item())
        q, mod = divmod(idx, 64)
        best_batch_in, best_batch_out = samples[q]
        best_in = best_batch_in[mod]
        best_out = best_batch_out[mod]
        best_in = best_in.view(1, 1, 28, 28).cpu()
        best_out = best_out.view(1, 1, 28, 28).cpu()
        grid_show(torch.cat((best_in, best_out), 3))
        plt.show()