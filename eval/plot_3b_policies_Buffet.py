import fnmatch
import itertools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

import args_parser


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def plot():

    """ Plot figures """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 24,
        "font.sans-serif": ["Helvetica"],
    })

    i = 1
    skip_n = 1

    games = ['Buffet', 'Buffet', 'Buffet', 'Buffet', ]
    variants = ["fp", "fp", "fp", "fp", ]
    num_disc_mfs = [120,] * 4
    major_obss = [0, 1, 5, 10]
    stationary = False

    for game, variant, num_disc_mf, major_obs in zip(games, variants, num_disc_mfs, major_obss):
        thickness = 1 / num_disc_mf

        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(3, 4, i)
        subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        color = clist.__next__()['color']
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        plot_vals = []
        plot_vals_2 = []

        config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                     num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary, temperature=0.1, softmax=0)
        files = find('stdout', config['exp_dir'])

        action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
        action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
        major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
        minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")

        env = config['game'](**config)
        point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2, 1 - 1 / config['num_disc_mf'] / 2, config['num_disc_mf'])] * (env.minor_observation_space.n - 1))
        points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
        mu_disc = np.array(points).transpose()
        

        plotted_mfs = mu_disc[1]
        # for mf_1 in range(len(plotted_mfs)).__reversed__():
        #     for t in range(env.time_steps):
        #         color = cmap(action_probs_major[t, major_obs, mf_1, 0])
        #         # plt.plot([t, t + 1], 2 * [plotted_mfs[mf_1]], color=color, label='_nolabel_',
        #         #          linewidth=2)
        #         plt.fill_between([t, t + 1], 2 * [plotted_mfs[mf_1] - thickness/2],
        #                          2 * [plotted_mfs[mf_1] + thickness/2], color=color)

        heatmap = np.zeros(len(plotted_mfs) * env.time_steps)
        idx = 0
        for mf_1 in range(len(plotted_mfs)).__reversed__():
            for t in range(env.time_steps):
                heatmap[idx] = action_probs_major[t, major_obs, mf_1, 1]
                idx += 1

        """ Plot """
        im = plt.imshow(heatmap.reshape((len(plotted_mfs), env.time_steps)), interpolation='none',
                        extent=[-0.5, env.time_steps - 0.5, 0, 1],
                        aspect=env.time_steps, cmap='hot_r', origin='lower', vmin=0, vmax=1)
        # plt.xticks(range(env.time_steps))
        plt.yticks([0, 0.5, 1])
        # plt.xlabel(r'State $x$')
        # plt.ylabel(r'Graphon index $\alpha$')

        # cb1 = plt.colorbar(im, fraction=0.046, pad=0.04)
        # # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
        # cb1.set_label(r'Final mean-field $\mu^\alpha_{49}(x)$')

        plt.title(f" $x^0=({env.to_fillings(major_obs)[0]}, {env.to_fillings(major_obs)[1]})$")
        # plt.grid('on')
        plt.xlabel(r'Time $t$', fontsize=22)
        if i == 2:
            plt.ylabel(r'MF $\mu(1)$', fontsize=22)
        else:
            frame1 = plt.gca()
            frame1.axes.yaxis.set_ticklabels([])

        if i == 5:
            cb1 = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb1.set_ticks([0, 0.5, 1], update_ticks=True)
            cb1.set_label(f"$\pi^0(u^0 = 1 \mid x^0, \mu)$")

            # divider = make_axes_locatable(plt.gca())
            # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
            # cb1.set_ticks([0, 1], update_ticks=True)
            # cb1.set_label(f"$\pi(1 \mid x, \mu)$")
            # plt.gcf().add_axes(ax_cb)

    for game, variant, num_disc_mf, major_obs in zip(games, variants, num_disc_mfs, major_obss):
        thickness = 1 / num_disc_mf

        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(3, 4, i)
        subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        color = clist.__next__()['color']
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        plot_vals = []
        plot_vals_2 = []

        config = args_parser.generate_config_from_kw(game=game, variant=variant,
                                                     fp_iterations=100 if variant == 'fpi' else 1000,
                                                     num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary,
                                                     temperature=0.1, softmax=0)
        files = find('stdout', config['exp_dir'])

        action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
        action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
        major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
        minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")

        env = config['game'](**config)
        point_candidates = itertools.product(
            *[np.linspace(1 / config['num_disc_mf'] / 2, 1 - 1 / config['num_disc_mf'] / 2, config['num_disc_mf'])] * (
                        env.minor_observation_space.n - 1))
        points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
        mu_disc = np.array(points).transpose()

        plotted_mfs = mu_disc[1]
        # for mf_1 in range(len(plotted_mfs)).__reversed__():
        #     for t in range(env.time_steps):
        #         color = cmap(action_probs_minor[t, 0, major_obs, mf_1, 0])
        #         # plt.plot([t, t + 1], 2 * [plotted_mfs[mf_1]], color=color, label='_nolabel_',
        #         #          linewidth=2)
        #         plt.fill_between([t, t + 1], 2 * [plotted_mfs[mf_1] - thickness/2],
        #                          2 * [plotted_mfs[mf_1] + thickness/2], color=color)

        heatmap = np.zeros(len(plotted_mfs) * env.time_steps)
        idx = 0
        for mf_1 in range(len(plotted_mfs)).__reversed__():
            for t in range(env.time_steps):
                heatmap[idx] = action_probs_minor[t, 0, major_obs, mf_1, 1]
                idx += 1

        """ Plot """
        im = plt.imshow(heatmap.reshape((len(plotted_mfs), env.time_steps)), interpolation='none',
                        extent=[-0.5, env.time_steps - 0.5, 0, 1],
                        aspect=env.time_steps, cmap='hot_r', origin='lower', vmin=0, vmax=1)
        # plt.xticks(range(env.time_steps))
        plt.yticks([0, 0.5, 1])
        # plt.xlabel(r'State $x$')
        # plt.ylabel(r'Graphon index $\alpha$')

        # cb1 = plt.colorbar(im, fraction=0.046, pad=0.04)
        # # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
        # cb1.set_label(r'Final mean-field $\mu^\alpha_{49}(x)$')

        plt.title(f" $x^0=({env.to_fillings(major_obs)[0]}, {env.to_fillings(major_obs)[1]})$")
        # plt.grid('on')
        plt.xlabel(r'Time $t$', fontsize=22)
        if i == 2+4:
            plt.ylabel(r'MF $\mu(1)$', fontsize=22)
        else:
            frame1 = plt.gca()
            frame1.axes.yaxis.set_ticklabels([])

        if i == 5+4:
            cb1 = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb1.set_ticks([0, 0.5, 1], update_ticks=True)
            cb1.set_label(f"$\pi(u=1 \mid x=1, x^0, \mu)$")

            # divider = make_axes_locatable(plt.gca())
            # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
            # cb1.set_ticks([0, 1], update_ticks=True)
            # cb1.set_label(f"$\pi(1 \mid x, \mu)$")
            # plt.gcf().add_axes(ax_cb)

    major_obss = [0, 5, 1, 2]

    for game, variant, num_disc_mf, major_obs in zip(games, variants, num_disc_mfs, major_obss):
        thickness = 1 / num_disc_mf

        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(3, 4, i)
        subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        color = clist.__next__()['color']
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        plot_vals = []
        plot_vals_2 = []

        config = args_parser.generate_config_from_kw(game=game, variant=variant,
                                                     fp_iterations=100 if variant == 'fpi' else 1000,
                                                     num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary,
                                                     temperature=0.1, softmax=0)
        files = find('stdout', config['exp_dir'])

        action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
        action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
        major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
        minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")

        env = config['game'](**config)
        point_candidates = itertools.product(
            *[np.linspace(1 / config['num_disc_mf'] / 2, 1 - 1 / config['num_disc_mf'] / 2, config['num_disc_mf'])] * (
                        env.minor_observation_space.n - 1))
        points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
        mu_disc = np.array(points).transpose()

        plotted_mfs = mu_disc[1]
        # for mf_1 in range(len(plotted_mfs)).__reversed__():
        #     for t in range(env.time_steps):
        #         color = cmap(action_probs_minor[t, 0, major_obs, mf_1, 0])
        #         # plt.plot([t, t + 1], 2 * [plotted_mfs[mf_1]], color=color, label='_nolabel_',
        #         #          linewidth=2)
        #         plt.fill_between([t, t + 1], 2 * [plotted_mfs[mf_1] - thickness/2],
        #                          2 * [plotted_mfs[mf_1] + thickness/2], color=color)

        heatmap = np.zeros(len(plotted_mfs) * env.time_steps)
        idx = 0
        for mf_1 in range(len(plotted_mfs)).__reversed__():
            for t in range(env.time_steps):
                heatmap[idx] = action_probs_minor[t, 1, major_obs, mf_1, 1]
                idx += 1

        """ Plot """
        im = plt.imshow(heatmap.reshape((len(plotted_mfs), env.time_steps)), interpolation='none',
                        extent=[-0.5, env.time_steps - 0.5, 0, 1],
                        aspect=env.time_steps, cmap='hot_r', origin='lower', vmin=0, vmax=1)
        # plt.xticks(range(env.time_steps))
        plt.yticks([0, 0.5, 1])
        # plt.xlabel(r'State $x$')
        # plt.ylabel(r'Graphon index $\alpha$')

        # cb1 = plt.colorbar(im, fraction=0.046, pad=0.04)
        # # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
        # cb1.set_label(r'Final mean-field $\mu^\alpha_{49}(x)$')

        plt.title(f" $x^0=({env.to_fillings(major_obs)[0]}, {env.to_fillings(major_obs)[1]})$")
        # plt.grid('on')
        plt.xlabel(r'Time $t$', fontsize=22)
        if i == 6+4:
            plt.ylabel(r'MF $\mu(1)$', fontsize=22)
        else:
            frame1 = plt.gca()
            frame1.axes.yaxis.set_ticklabels([])

        if i == 9+4:
            cb2 = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb2.set_ticks([0, 0.5, 1], update_ticks=True)
            cb2.set_label(f"$\pi(u=1 \mid x=0, x^0, \mu)$")

            # divider = make_axes_locatable(plt.gca())
            # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
            # cb1.set_ticks([0, 1], update_ticks=True)
            # cb1.set_label(f"$\pi(1 \mid x, \mu)$")
            # plt.gcf().add_axes(ax_cb)

    """ Finalize plot """
    plt.gcf().set_size_inches(16, 12.75)
    plt.tight_layout(w_pad=0.0)
    plt.savefig(f'./figures/policies_Buffet.pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.savefig(f'./figures/policies_Buffet.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    plot()
