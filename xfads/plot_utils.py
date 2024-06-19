import torch
import numpy as np


def plot_two_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=500, device='cpu'):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        X, Y = np.meshgrid(x, y)

        XY = torch.zeros((X.shape[0]**2, 2))
        XY[:, 0] = torch.from_numpy(X).flatten().to(device)
        XY[:, 1] = torch.from_numpy(Y).flatten().to(device)

        XY_out = dynamics_fn(XY)
        s = XY_out - XY
        u = s[:, 0].reshape(X.shape[0], X.shape[1])
        v = s[:, 1].reshape(Y.shape[0], Y.shape[1])

        axs.streamplot(X, Y, u, v, color='black', linewidth=0.5, arrowsize=0.5)


def plot_z_samples(fig, axs, samples, color_map_list):
    n_samples, n_trials, n_bins, n_neurons = samples.shape
    fig.subplots_adjust(hspace=0)
    [axs[i].axvline(12, linestyle='--', color='gray') for i in range(n_trials)]
    [axs[i].axis('off') for i in range(n_trials)]
    [axs[i].plot(samples[j, i, :, n], color=color_map_list[n](j), linewidth=0.5, alpha=0.4)
     for i in range(n_trials) for j in range(samples.shape[0]) for n in range(n_neurons)]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    [axs[i].set_ylim(-10, 10) for i in range(n_trials)]
    fig.tight_layout()


def plot_spikes(spikes, axs):
    n_bins = spikes.shape[0]
    n_neurons = spikes.shape[1]

    # fig, axs = plt.subplots(figsize=(6, 3))
    _, indices = torch.sort(spikes.mean(dim=0))
    spikes = spikes[:, indices][..., n_neurons//2:]
    n_neurons = n_neurons//2

    for n in range(n_neurons):
        time_ax = np.arange(n_bins)
        neuron_spikes = spikes[:, n]
        neuron_spikes[neuron_spikes > 0] = 1
        neuron_spikes = neuron_spikes * time_ax
        neuron_spikes = neuron_spikes[neuron_spikes > 0]

        axs.scatter(neuron_spikes, 0.5 * n * np.ones_like(neuron_spikes), marker='o', color='black', s=4,
                    edgecolors='none')