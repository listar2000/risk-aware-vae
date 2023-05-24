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


def generate_img(model, z_dim, device):
    """
    Generate a random sample of images from a VAE model
    """

    with torch.no_grad():
        z = torch.randn(model.batch_size, z_dim).unsqueeze(-1).to(device)
        sample = model.model.decode(z)
    return sample.cpu()


def reconstruct_img(model, x, device):
    """
    Reconstruct a image by doing one forward pass of a VAE model
    """ 

    with torch.no_grad():
        sample, _, _ = model.model.evaluate(x.to(device))
    return sample.cpu()


def show_gen_img(model, z_dim, device):
    sample = generate_img(model, z_dim, device)
    sample = sample.view(model.batch_size, 1, 28, 28)
    grid_show(sample)


def show_recon_img(model, x, device):
    sample = reconstruct_img(model, x, device)
    imgs = torch.cat((x.view(1, 1, 28, 28).cpu(), sample.view(1, 1, 28, 28)))
    grid_show(imgs)


def generated_embedding(model, z_dim, device, loader):
    """
    Extract embedding from data in a dataloader. Additionally generate batches 
    of images and extract their embedings by passing it through the encoder one 
    more time.
    """

    list_samples = []
    list_mu_real = []
    list_mu_gen = []
    with torch.no_grad():
        for data in tqdm(loader):
            data = data[0].to(device)
            data = data.view(data.size(0), -1)
            mu_real = model.model.encode(data)[0]
            list_mu_real.append(mu_real)
            gen_img = generate_img(model, z_dim, device)
            gen_img = gen_img.view(gen_img.shape[0], -1).to(device)
            mu_gen = model.model.encode(gen_img)[0]
            list_mu_gen.append(mu_gen)
            list_samples.append(gen_img)
    return (list_samples, list_mu_real, list_mu_gen)


def dist(list_mu_a, list_mu_b):
    """
    Compute pairwise distances in feature space.
    """
    
    list_Mxx = []
    list_Mxy = []
    list_Myy = []
    for mu_a, mu_b in zip(list_mu_a, list_mu_b):
        list_Mxx.append(torch.cdist(mu_a, mu_a).cpu())
        list_Mxy.append(torch.cdist(mu_a, mu_b).cpu())
        list_Myy.append(torch.cdist(mu_b, mu_b).cpu())
    return (list_Mxx, list_Mxy, list_Myy)


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


def knn_by_batch(list_mu_a, list_mu_b):
    list_Mxx, list_Mxy, list_Myy = dist(list_mu_a, list_mu_b)
    list_acc = []
    for Mxx, Mxy, Myy in zip(list_Mxx, list_Mxy, list_Myy):
        list_acc.append(knn(Mxx, Mxy, Myy)[0])
    return list_acc


def mmd(Mxx, Mxy, Myy, sigma=1):
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


def mmd_by_batch(list_mu_a, list_mu_b):
    list_Mxx, list_Mxy, list_Myy = dist(list_mu_a, list_mu_b)
    list_mmd = []
    for Mxx, Mxy, Myy in zip(list_Mxx, list_Mxy, list_Myy):
        list_mmd.append(mmd(Mxx, Mxy, Myy))
    return list_mmd


def plot_metric_best_batch(samples, list_metric):
    best_batch = samples[np.argmin(list_metric)]
    best_batch = best_batch.view(64, 1, 28, 28).cpu()
    grid_show(best_batch)
    plt.show()


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


def compute_recon_loss(model, val_dataloader, device):
    """
    Given a validation set, get all reconstructed samples and associated losses
    """
    
    list_recon_loss = []
    list_recon_samples = []
    for val_features, _ in tqdm(val_dataloader):
        b = val_features.shape[0]
        val_samples = val_features.view(b, -1)
        recon_samples = reconstruct_img(model, val_samples, device)
        list_recon_samples.append((val_samples, recon_samples))
        val_samples_expand = val_samples.unsqueeze(1).expand_as(recon_samples)
        recon_loss = model.recon_loss_f(recon_samples.to(device), val_samples_expand.to(device), reduction="none")
        list_recon_loss.append(recon_loss.sum(-1).cpu())

    return list_recon_samples, list_recon_loss


def reorder(list_recon_loss, by):
    """
    Reshape the list of reconstruction loss
    """
    
    if by == "image":
        # loss by sub-sample
        return torch.vstack(list_recon_loss).flatten().tolist()
    elif by == "image_nested":
        # loss by sub-sample, organized in batches
        return [loss.flatten().tolist() for loss in list_recon_loss]
    elif by == "batch":
        # loss by batch average
        return [loss.mean().item() for loss in list_recon_loss]


def plot_metric(metric_a, metric_b, metric):
    """
    Plot reconstruction loss as histograms
    """
    
    a_min = np.min(metric_a)
    a_max = np.max(metric_a)
    b_min = np.min(metric_b)
    b_max = np.max(metric_b)
    print(f"Model A: {metric} minimum: {a_min}, {metric} maximum: {a_max}")
    print(f"Model B: {metric} minimum: {b_min}, {metric} maximum: {b_max}")
    plt.hist(metric_a, density=True, histtype='step')
    plt.hist(metric_b, density=True, histtype='step')
    plt.show()
            

def best_batch(samples, loss_by_batch, loss_by_image_nested):
    print("Loss on best batch:", np.min(loss_by_batch))
    best_idx = np.argmin(loss_by_batch)
    return (samples[best_idx], loss_by_image_nested[best_idx])


def plot_best_batch(best_batch):
    """
    From the best batch, plot a randomly chosen batch.
    """
    
    best_batch_in, best_batch_out = best_batch
    random_idx = torch.randint(best_batch_out.shape[1] - 1, (1,))
    best_batch_out = best_batch_out[:, random_idx, :]
    best_batch_in = best_batch_in.view(64, 1, 28, 28).cpu()
    best_batch_out = best_batch_out.view(64, 1, 28, 28).cpu()
    grid_show(torch.cat((best_batch_in, best_batch_out), 3))
    plt.show()


def best_images(samples, loss, n=10):
    best_images_idx = np.argsort(loss)[:n]
    best_loss = [loss[idx] for idx in best_images_idx]
    best_images = []
    for idx in best_images_idx:
        print("Reconstruction loss:", loss[idx])
        q, mod = divmod(idx, 6400)
        row_idx, col_idx = divmod(mod, 100)
        best_batch_in, best_batch_out = samples[q]
        best_in = best_batch_in[row_idx]
        best_out = best_batch_out[row_idx, col_idx]
        best_images.append((best_in, best_out))

    return (best_images, best_loss)


def plot_best_images(best_images, n=3):
    for i in range(n):
        best_in, best_out = best_images[i]
        best_in = best_in.view(1, 1, 28, 28).cpu()
        best_out = best_out.view(1, 1, 28, 28).cpu()
        grid_show(torch.cat((best_in, best_out), 3))
        plt.show()


def t_test(recon_loss_a, recon_loss_b):
    t, p = stats.ttest_ind(recon_loss_a, recon_loss_b)
    conclusion = f"difference, t-statistics: {t}, p-value: {p}"
    conclusion = "Significant " + conclusion if p < 0.05 else "No significant " + conclusion
    print(conclusion)
