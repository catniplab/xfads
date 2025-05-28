import torch
import numpy as np
import einops
import matplotlib.pyplot as plt

from matplotlib import cm

# from pyrecorder.recorder import Recorder
# from pyrecorder.writers.video import Video
# from pyrecorder.converters.matplotlib import Matplotlib
from matplotlib.animation import FFMpegWriter


def plot_two_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=500):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        X, Y = np.meshgrid(x, y)

        XY = torch.zeros((X.shape[0] ** 2, 2))
        XY[:, 0] = torch.from_numpy(X).flatten()
        XY[:, 1] = torch.from_numpy(Y).flatten()

        XY_out = dynamics_fn(XY)
        s = XY_out - XY
        u = s[:, 0].reshape(X.shape[0], X.shape[1])
        v = s[:, 1].reshape(Y.shape[0], Y.shape[1])

        axs.streamplot(X, Y, u, v, color="black", linewidth=0.5, arrowsize=0.5)


def plot_z_samples(fig, axs, samples, color_map_list):
    n_samples, n_trials, n_bins, n_neurons = samples.shape
    fig.subplots_adjust(hspace=0)
    [axs[i].axvline(12, linestyle="--", color="gray") for i in range(n_trials)]
    [axs[i].axis("off") for i in range(n_trials)]
    [
        axs[i].plot(
            samples[j, i, :, n], color=color_map_list[n](j), linewidth=0.5, alpha=0.4
        )
        for i in range(n_trials)
        for j in range(samples.shape[0])
        for n in range(n_neurons)
    ]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    [axs[i].set_ylim(-20, 28) for i in range(n_trials)]
    fig.tight_layout()


def plot_samples_w_input(axs, samples, inputs, color_map_list):
    n_inputs = inputs.shape[-1]
    n_samples, n_trials, n_bins, n_neurons = samples.shape
    trial_dx_time_dx = [torch.where(inputs[..., k] == 1) for k in range(n_inputs)]

    # [axs[i].axvline(12, linestyle='--', color='gray') for i in range(n_trials)]
    for input_dx in range(n_inputs):
        [
            axs[trial_dx_time_dx[input_dx][0][i]].axvline(
                trial_dx_time_dx[input_dx][1][i], linestyle="--", color="gray"
            )
            for i in range(trial_dx_time_dx[input_dx][0].shape[0])
        ]

    [axs[i].axis("off") for i in range(n_trials)]
    [
        axs[i].plot(
            samples[j, i, :, n], color=color_map_list[n](j), linewidth=0.5, alpha=0.4
        )
        for i in range(n_trials)
        for j in range(samples.shape[0])
        for n in range(n_neurons)
    ]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    [axs[i].set_ylim(-10, 10) for i in range(n_trials)]


def plot_spikes(spikes, axs):
    n_bins = spikes.shape[0]
    n_neurons = spikes.shape[1]

    # fig, axs = plt.subplots(figsize=(6, 3))
    _, indices = torch.sort(spikes.mean(dim=0))
    spikes = spikes[:, indices][..., n_neurons // 2 :]
    n_neurons = n_neurons // 2

    for n in range(n_neurons):
        time_ax = np.arange(n_bins)
        neuron_spikes = spikes[:, n]
        neuron_spikes[neuron_spikes > 0] = 1
        neuron_spikes = neuron_spikes * time_ax
        neuron_spikes = neuron_spikes[neuron_spikes > 0]

        # axs.scatter(neuron_spikes, 0.5 * n * np.ones_like(neuron_spikes), marker='o', color='black', s=4,
        #             edgecolors='none')
        axs.scatter(
            neuron_spikes,
            0.5 * n * np.ones_like(neuron_spikes),
            marker="o",
            color="black",
            s=0.1,
            edgecolors="none",
        )


def plot_reaching(axs, pos, reach_colors):
    n_trials, n_bins, _ = pos.shape
    axs.axis("off")

    for n in range(n_trials):
        axs.plot(pos[n, :, 0], pos[n, :, 1], color=reach_colors[n])


def plot_single_trial_reaching_latents(z):
    ylim = torch.abs(z).amax(dim=[0, 1])  # max per trial
    n_samples, n_bins, n_latents = z.shape

    blues = cm.get_cmap("Blues", n_samples)
    # grays = cm.get_cmap("Greys", n_samples)

    with torch.no_grad():
        fig, ax = plt.subplots(n_latents // 4, 4, figsize=(4, 4))
        ax = ax.flatten()
        fig.subplots_adjust(hspace=0.0)
        fig.subplots_adjust(wspace=0.0)
        [ax[n].axis("off") for n in range(n_latents)]

        for n in range(n_latents):
            [
                ax[n].plot(z[s, :, n], color=blues(s), linewidth=0.1)
                for s in range(n_samples)
            ]
            ax[n].set_ylim((-ylim[n], ylim[n]))
            ax[n].set_xlim((0, n_bins))

        fig.tight_layout()

    return fig


# def animate_reaching(pos, reach_colors, animation_path):
#
#     writer = Video(animation_path, fps=10)
#     converter = Matplotlib(dpi=250)
#     rec = Recorder(writer, converter=converter)
#     # xlim = torch.abs(pos[:, :, 0]).max()
#     # ylim = torch.abs(pos[:, :, 1]).max()
#     xlim = 7500
#     ylim = 7500
#     n_trials, n_bins, _ = pos.shape
#
#     for t in range(n_bins):
#         fig, ax = plt.subplots(1, figsize=(3, 3))
#         ax.set_box_aspect(1.0)
#         ax.axis('off')
#
#         for n in range(n_trials):
#             ax.plot(pos[n, :t, 0], pos[n, :t, 1], color=reach_colors[n])
#             ax.set_xlim((-xlim, xlim))
#             ax.set_ylim((-ylim, ylim))
#
#         fig.tight_layout()
#         rec.record(fig=fig)
#     rec.close()
#
#     return fig


# [plot_utils.plot_reaching(axs[4 + k], pos_test_hat_p[k][trial_plt_dx], reach_colors[trial_plt_dx]) for k in
#  range(predict_start_max)]


def animate_reaching_evolution(
    r2_prior,
    r2_prd,
    r2_f,
    pos_prior,
    pos_prd,
    pos_f,
    reach_colors,
    trial_plt_dx,
    speeds_gt,
    speeds_hat,
    animation_path,
):
    xlim = 7500
    ylim = 7500
    n_trials, n_bins, _ = pos_prior.shape
    n_start_pts = len(pos_prd)

    speed_std = [torch.cat(speeds_hat[k], dim=0).std(dim=0) for k in range(n_start_pts)]
    speed_gt_std = torch.cat(speeds_gt, dim=0).std(dim=0)
    speed_mean = [
        torch.cat(speeds_hat[k], dim=0).mean(dim=0) for k in range(n_start_pts)
    ]
    speed_gt_mean = torch.cat(speeds_gt, dim=0).mean(dim=0)

    writer = FFMpegWriter(fps=10)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax = ax.flatten()
    ax[1].set_xlim((-xlim, xlim))
    ax[1].set_ylim((-ylim, ylim))
    ax[1].set_box_aspect(1.0)
    ax[1].axis("off")

    with writer.saving(fig, "plots/animation.mp4", 300):
        [
            ax[0].plot(pos_f[n, :, 0], pos_f[n, :, 1], color=reach_colors[n])
            for n in trial_plt_dx
        ]
        ax[0].set_title(f"R2: {r2_f:.2f}")
        ax[0].set_xlim((-xlim, xlim))
        ax[0].set_ylim((-ylim, ylim))
        ax[0].set_box_aspect(1.0)
        ax[0].axis("off")

        for t in range(0, n_start_pts):
            ax[1].set_title(f"R2: {r2_prd[t]:.2f},  -{240 - 20 * t}ms")

            for s in range(speed_gt_mean.shape[0]):
                for n in trial_plt_dx:
                    ax[1].plot(
                        pos_prd[t][n, :s, 0],
                        pos_prd[t][n, :s, 1],
                        color=reach_colors[n],
                    )

                ax[2].clear()
                x_vals = -240 + 20 * np.arange(35)
                ax[2].plot(x_vals, speed_mean[t], color="red", label="model")
                ax[2].axvline(x_vals[12], linestyle="--", color="grey", alpha=0.7)
                ax[2].axvline(x_vals[t], linestyle="--", color="red", alpha=0.4)
                ax[2].plot(x_vals, speed_gt_mean, color="black", label="true")
                ax[2].fill_between(
                    x_vals,
                    speed_mean[t] + 2 * speed_std[t],
                    speed_mean[t] - 2 * speed_std[t],
                    color="red",
                    linewidth=0.0,
                    alpha=0.2,
                )
                # ax[2].plot(x_vals, speed_mean[t] - 2 * speed_std[t], color='red', linestyle='--')
                ax[2].fill_between(
                    x_vals,
                    speed_gt_mean + 2 * speed_gt_std,
                    speed_gt_mean - 2 * speed_gt_std,
                    color="black",
                    linewidth=0.0,
                    alpha=0.2,
                )
                ax[2].plot(
                    x_vals,
                    speed_gt_mean - 2 * speed_gt_std,
                    color="black",
                    linestyle="--",
                )
                ax[2].scatter(x_vals[s], speed_mean[t][s], color="red")
                ax[2].set_box_aspect(1.0)
                ax[2].set_xlabel("time (ms)")
                ax[2].set_ylim(0, 1200)
                ax[2].set_title("speed (a.u)")
                ax[2].legend()
                # labels = ax[2].get_xticks().tolist()
                # labels[0] = '-240'
                # labels[12] = '0'
                # labels[-1] = '460'
                # ax[2].set_xticklabels(labels)
                labels = ax[2].get_yticks().tolist()
                labels = ["" for x in labels]
                labels[-2] = "1"
                labels[0] = "0"
                ax[2].set_yticklabels(labels)

                writer.grab_frame()
                # ax[2].clear()

            writer.grab_frame()

            fig.savefig(
                f"plots/animation_t_{t}.pdf", transparent=True, bbox_inches="tight"
            )
            if t != n_start_pts - 1:
                ax[1].clear()
                ax[2].clear()

            ax[1].set_xlim((-xlim, xlim))
            ax[1].set_ylim((-ylim, ylim))
            ax[1].set_box_aspect(1.0)
            ax[1].axis("off")

        # writer.grab_frame()
        # fig.clear()
        # plot_reaching(ax, pos_prd[t][trial_plt_dx], reach_colors[trial_plt_dx])
        # ax.set_xlim((-xlim, xlim))
        # ax.set_ylim((-ylim, ylim))
        # ax.set_title(f'R2: {r2_prd[t]}')

        # for n in range(n_trials):
        #     ax.plot(pos[n, :t, 0], pos[n, :t, 1], color=reach_colors[n])
        #     ax.set_xlim((-xlim, xlim))
        #     ax.set_ylim((-ylim, ylim))

        # fig.tight_layout()

    return fig


def animate_reaching_evolution_mo(
    r2_prd,
    r2_f,
    pos_prd,
    pos_f,
    reach_colors,
    trial_plt_dx,
    speeds_gt,
    speeds_hat,
    avg_position,
):
    # def animate_reaching_evolution_mo(r2_prd, r2_f, pos_prd, pos_f, reach_colors, trial_plt_dx, speeds_hat):
    xlim = 7500
    ylim = 7500
    # n_trials, n_bins, _ = pos_prior.shape
    n_trials, n_bins, _ = pos_prd[0].shape
    n_start_pts = len(pos_prd)

    speed_std = [speeds_hat[k].mean(dim=0).std(dim=0) for k in range(len(speeds_hat))]
    speed_mean = [speeds_hat[k].mean(dim=0).mean(dim=0) for k in range(len(speeds_hat))]
    speed_gt_std = speeds_gt.std(dim=0)
    speed_gt_mean = speeds_gt.mean(dim=0)

    writer = FFMpegWriter(fps=10)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax = [ax[1], ax[2], ax[3], ax[0]]
    ax[1].set_xlim((-xlim, xlim))
    ax[1].set_ylim((-ylim, ylim))
    ax[1].set_box_aspect(1.0)
    ax[1].axis("off")
    ax[-1].set_xlim((-xlim, xlim))
    ax[-1].set_ylim((-ylim, ylim))
    ax[-1].set_box_aspect(1.0)
    ax[-1].axis("off")

    for i in range(len(avg_position["trajectory"])):
        ax[-1].plot(
            avg_position["trajectory"][i][:, 0],
            avg_position["trajectory"][i][:, 1],
            color=avg_position["color"][i],
        )

    with writer.saving(fig, "plots/animation.mp4", 300):
        [
            ax[0].plot(pos_f[n, :, 0], pos_f[n, :, 1], color=reach_colors[n])
            for n in trial_plt_dx
        ]
        ax[0].set_title(f"R2: {r2_f:.2f}")
        ax[0].set_xlim((-xlim, xlim))
        ax[0].set_ylim((-ylim, ylim))
        ax[0].set_box_aspect(1.0)
        ax[0].axis("off")

        for t in range(0, n_start_pts):
            if 240 - 20 * t > 0:
                ax[1].set_title(f"R2: {r2_prd[t]:.2f},  -{240 - 20 * t}ms")
            else:
                ax[1].set_title(f"R2: {r2_prd[t]:.2f},  {240 - 20 * t}ms")

            # for s in range(speed_gt_mean.shape[0]):
            for s in range(pos_prd[t].shape[1]):
                for n in trial_plt_dx:
                    ax[1].plot(
                        pos_prd[t][n, :s, 0],
                        pos_prd[t][n, :s, 1],
                        color=reach_colors[n],
                    )
                    if t > 0 and s == 0:
                        ax[1].plot(
                            pos_prd[t - 1][n, :, 0],
                            pos_prd[t - 1][n, :, 1],
                            color=reach_colors[n],
                            alpha=0.25,
                        )

                ax[2].clear()
                x_vals = -700 + 20 * np.arange(speed_mean[0].shape[0])
                ax[2].plot(x_vals, speed_mean[t], color="red", label="model")
                ax[2].axvline(x_vals[35], linestyle="--", color="grey", alpha=0.7)
                ax[2].axvline(x_vals[t], linestyle="--", color="red", alpha=0.4)
                ax[2].plot(x_vals, speed_gt_mean, color="black", label="true")
                ax[2].fill_between(
                    x_vals,
                    (speed_mean[t] + 2 * speed_std[t]).squeeze(-1),
                    (speed_mean[t] - 2 * speed_std[t]).squeeze(-1),
                    color="red",
                    linewidth=0.0,
                    alpha=0.2,
                )
                # ax[2].plot(x_vals, speed_mean[t] - 2 * speed_std[t], color='red', linestyle='--')
                ax[2].fill_between(
                    x_vals,
                    (speed_gt_mean + 2 * speed_gt_std).squeeze(-1),
                    (speed_gt_mean - 2 * speed_gt_std).squeeze(-1),
                    color="black",
                    linewidth=0.0,
                    alpha=0.2,
                )
                # ax[2].plot(x_vals, speed_gt_mean - 2 * speed_gt_std, color='black', linestyle='--')
                ax[2].scatter(x_vals[s], speed_mean[t][s], color="red")
                ax[2].set_box_aspect(1.0)
                ax[2].set_xlabel("time (ms)")
                ax[2].set_ylim(0, 1200)
                ax[2].set_title("speed (a.u)")
                ax[2].legend()
                # labels = ax[2].get_xticks().tolist()
                # labels[0] = '-240'
                # labels[12] = '0'
                # labels[-1] = '460'
                # ax[2].set_xticklabels(labels)
                labels = ax[2].get_yticks().tolist()
                labels = ["" for x in labels]
                labels[-2] = "1"
                labels[0] = "0"
                ax[2].set_yticklabels(labels)

                writer.grab_frame()
                # ax[2].clear()

            writer.grab_frame()

            fig.savefig(
                f"plots/animation_t_{t}.pdf", transparent=True, bbox_inches="tight"
            )
            if t != n_start_pts - 1:
                ax[1].clear()
                ax[2].clear()

            ax[1].set_xlim((-xlim, xlim))
            ax[1].set_ylim((-ylim, ylim))
            ax[1].set_box_aspect(1.0)
            ax[1].axis("off")

        # writer.grab_frame()
        # fig.clear()
        # plot_reaching(ax, pos_prd[t][trial_plt_dx], reach_colors[trial_plt_dx])
        # ax.set_xlim((-xlim, xlim))
        # ax.set_ylim((-ylim, ylim))
        # ax.set_title(f'R2: {r2_prd[t]}')

        # for n in range(n_trials):
        #     ax.plot(pos[n, :t, 0], pos[n, :t, 1], color=reach_colors[n])
        #     ax.set_xlim((-xlim, xlim))
        #     ax.set_ylim((-ylim, ylim))

        # fig.tight_layout()

    return fig


def plot_ball(y, y_hat, z_hat, w=16, label_dx=None):
    n_time_bins, n_neurons = y.shape

    with torch.no_grad():
        fig, axs = plt.subplots(3, n_time_bins, figsize=(n_time_bins, 3))
        gs = axs[-1, 0].get_gridspec()
        # remove the underlying axes
        for ax in axs[-1, :]:
            ax.remove()

        axbig = fig.add_subplot(gs[-1, :])
        y_img = einops.rearrange(y, "t (w h) -> t w h", w=w, h=w)
        y_img[y_img < 0.7] = 0.0
        y_hat_img = einops.rearrange(y_hat, "t (w h) -> t w h", w=w, h=w)

        for t in range(n_time_bins):
            if t == label_dx:
                axs[0, t].set_title("*")

            axs[0, t].imshow(y_img[t], cmap="gray_r")  # , vmin=0, vmax=255)
            axs[1, t].imshow(y_hat_img[t], cmap="gray_r")  # , vmin=0, vmax=255)
            axs[0, t].tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )
            axs[1, t].tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )
            # axs[0, t].axis('off')
            # axs[1, t].axis('off')

            for location in ["left", "right", "top", "bottom"]:
                axs[0, t].spines[location].set_linewidth(0.1)
                axs[1, t].spines[location].set_linewidth(0.1)

            # axs[2, t].tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)

        fig.suptitle("reconstructions")
        axs[0, 0].set_ylabel("true")
        axs[1, 0].set_ylabel("sample")
        # axs[2, 0].set_ylabel('noisy')
        [
            axbig.plot(np.arange(n_time_bins), z_hat[..., i])
            for i in range(z_hat.shape[-1])
        ]
        axbig.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)

        if label_dx is not None:
            axbig.axvline(x=label_dx, linestyle="--")

    return fig


def plot_ball_covariance(y, P_diag, P=None, w=16, label_dx=None):
    max_P = P_diag.max()
    min_P = P_diag.min()
    n_time_bins, n_neurons = y.shape

    with torch.no_grad():
        fig, axs = plt.subplots(3, n_time_bins, figsize=(n_time_bins, 3))

        y_img = einops.rearrange(y, "t (w h) -> t w h", w=w, h=w)
        y_img[y_img < 0.7] = 0.0

        for t in range(n_time_bins):
            if t == label_dx:
                axs[0, t].set_title("*")

            axs[0, t].imshow(y_img[t], cmap="gray_r")  # , vmin=0, vmax=255)
            axs[2, t].imshow(
                torch.diag(P_diag[t]), cmap="gray_r", vmin=min_P, vmax=max_P
            )  # , vmin=0, vmax=255)
            if P is not None:
                axs[1, t].imshow(
                    P[t], cmap="gray_r", vmin=min_P, vmax=max_P
                )  # , vmin=0, vmax=255)

            axs[0, t].tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )
            axs[1, t].tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )
            axs[2, t].tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )

            for location in ["left", "right", "top", "bottom"]:
                axs[0, t].spines[location].set_linewidth(0.1)
                axs[1, t].spines[location].set_linewidth(0.1)
                axs[2, t].spines[location].set_linewidth(0.1)

        fig.suptitle("covariance over time")
        axs[0, 0].set_ylabel("true")
        axs[1, 0].set_ylabel("P")
        axs[2, 0].set_ylabel("P_diag")
        # axbig.set_ylabel('P_diag')
        # [axbig.plot(np.arange(n_time_bins), P_diag[..., i]) for i in range(P_diag.shape[-1])]
        # axbig.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)

        # if label_dx is not None:
        #     axbig.axvline(x=label_dx, linestyle='--')

    return fig
