import torch
from dataset import read_mnist
from backbone import VAE, two_layer_config

if __name__ == '__main__':
    mnist_train, mnist_val, mnist_test = read_mnist()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ]
    ) as p:
        vae_model = VAE(28 * 28, 20, two_layer_config, device=device, risk_aware="neutral")
        vae_model.fit(mnist_train, mnist_test, epochs=5)

    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
