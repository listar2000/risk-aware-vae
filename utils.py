import matplotlib.pyplot as plt
import torch


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
