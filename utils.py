import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from backbone import VAE, two_layer_config
from dataset import read_mnist


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
    import numpy as np
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="gray")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def generate_img(model, z_dim, device):
    with torch.no_grad():
        z = torch.randn(64, z_dim).unsqueeze(-1).to(device)
        sample = model.model.decode(z)
    img_grid = make_grid(sample.view(64, 1, 28, 28).cpu())
    matplotlib_imshow(img_grid, one_channel=True)


def reconstruct_img(model, x, device):
    with torch.no_grad():
        sample, _, _ = model.model.evaluate(x.reshape(1, -1).to(device))

    imgs = torch.cat((x.view(1, 1, 28, 28).cpu(), sample.view(1, 1, 28, 28).cpu()))
    img_grid = make_grid(imgs)
    matplotlib_imshow(img_grid, one_channel=True)
    plt.show()


def train_mnist(z_dim, config, device, risk_aware, epochs=10, risk_q=0.5, show_config=True, plot=True):
    mnist_train, mnist_val, mnist_test = read_mnist()
    train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    model = VAE(28 * 28, z_dim, config, device=device, risk_aware=risk_aware, risk_q=risk_q)
    if show_config:
        print(model.model)
    model.fit(mnist_train, mnist_val, epochs=epochs)
    if plot:
        plt.subplot(1, 2, 1)
        generate_img(model, z_dim, device)
        plt.subplot(1, 2, 2)
        reconstruct_img(model, train_features[0])
        plt.show()
    return model


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae_model = train_mnist(20, two_layer_config, device, risk_aware="neutral", plot=False)
