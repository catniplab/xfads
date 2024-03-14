import torch
import numpy as np


def plot_two_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=100, device='cpu'):
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
