import fnmatch
import itertools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import MaxNLocator

import args_parser
from utils import get_softmax_action_probs_from_Qs, eval_curr_major_reward, eval_curr_minor_reward


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

    games = ['SIS', 'Buffet', 'Advertisement']
    variants = ["fpi",] * len(games)
    stationary = False

    for game, variant in zip(games, variants):
        num_disc_mfs = range(10, 205, 10) if game != 'Buffet' else range(10, 135, 10)

        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(2, len(games), i)
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
                         (1, 1),
                         xytext=(-36, -10),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        plot_vals = []
        plot_vals_2 = []

        for num_disc_mf in num_disc_mfs:
            config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=1,
                                                         num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary, softmax=True)

            env = config['game'](**config)

            """ Epsilon net of simplex for mean field discretization """
            point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2,
                                                               1 - 1 / config['num_disc_mf'] / 2,
                                                               config['num_disc_mf'])] * (
                                                              env.minor_observation_space.n - 1))
            points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
            mu_disc = np.array(points).transpose()
            dists_to_net = np.sum(np.abs(np.expand_dims(env.mu_0, axis=-1) - mu_disc), axis=0)
            mu_index_initial = dists_to_net.argmin()

            major_Q_0 = [np.zeros(
                (env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], env.major_action_space.n))]
            minor_Q_0 = [np.zeros((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n,
                                   mu_disc.shape[-1], env.minor_action_space.n))]

            action_probs_minor = get_softmax_action_probs_from_Qs(np.array(minor_Q_0),
                                                                  temperature=config['temperature'])
            action_probs_major = get_softmax_action_probs_from_Qs(np.array(major_Q_0),
                                                                  temperature=config['temperature'])

            v_curr_1 = np.vdot(env.mu_0_major,
                               eval_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major)[0][:,
                               mu_index_initial])
            v_curr_2 = np.vdot(env.mu_0, np.einsum('i,ji', env.mu_0_major,
                                                   eval_curr_minor_reward(env, mu_disc, action_probs_minor,
                                                                          action_probs_major)[0][:, :,
                                                   mu_index_initial]))

            plot_vals.append(v_curr_1)
            plot_vals_2.append(v_curr_2)
            print(fr"{num_disc_mf}: {plot_vals_2}")

        plot_vals = np.array(plot_vals)
        plot_vals_2 = np.array(plot_vals_2)

        color = clist.__next__()['color']
        # linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_disc_mfs[::skip_n], plot_vals[::skip_n], linestyle, color=color,
        #              label="$J^0$", alpha=0.5)
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(num_disc_mfs[::skip_n], plot_vals_2[::skip_n], linestyle, color=color,
                     label="$J$", alpha=0.5)
        subplot.scatter(num_disc_mfs[::skip_n], plot_vals_2[::skip_n], color=color,
                     label="$J$", alpha=0.5)
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot([min(num_disc_mfs), max(num_disc_mfs)], [plot_vals_2[-1]] * 2, linestyle, color=color,
                     label="$J^0$", alpha=0.5)
        subplot.scatter([min(num_disc_mfs), max(num_disc_mfs)], [plot_vals_2[-1]] * 2, color=color,
                     label="$J^0$", alpha=0.5)

        # if i==7:
        #     plt.legend()
        # plt.title(game)
        plt.grid('on')
        # plt.xlabel(r'MF grid fineness $M$', fontsize=22)
        if i==2:
            plt.ylabel(r'$\hat J$', fontsize=22)
        plt.xlim([min(num_disc_mfs), max(num_disc_mfs)])
        # plt.ylim([0, 500])

    for game, variant in zip(games, variants):
        num_disc_mfs = range(10, 205, 10) if game != 'Buffet' else range(10, 135, 10)

        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(2, len(games), i)
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
                         (1, 1),
                         xytext=(-36, -10 if i != 6 else -95),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        plot_vals = []
        plot_vals_2 = []

        for num_disc_mf in num_disc_mfs:
            config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=1,
                                                         num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary,
                                                         softmax=True)

            env = config['game'](**config)

            """ Epsilon net of simplex for mean field discretization """
            point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2,
                                                               1 - 1 / config['num_disc_mf'] / 2,
                                                               config['num_disc_mf'])] * (
                                                          env.minor_observation_space.n - 1))
            points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
            mu_disc = np.array(points).transpose()
            dists_to_net = np.sum(np.abs(np.expand_dims(env.mu_0, axis=-1) - mu_disc), axis=0)
            mu_index_initial = dists_to_net.argmin()

            major_Q_0 = [np.zeros(
                (env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], env.major_action_space.n))]
            minor_Q_0 = [np.zeros((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n,
                                   mu_disc.shape[-1], env.minor_action_space.n))]

            action_probs_minor = get_softmax_action_probs_from_Qs(np.array(minor_Q_0),
                                                                  temperature=config['temperature'])
            action_probs_major = get_softmax_action_probs_from_Qs(np.array(major_Q_0),
                                                                  temperature=config['temperature'])

            v_curr_1 = np.vdot(env.mu_0_major,
                               eval_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major)[0][:,
                               mu_index_initial])
            v_curr_2 = np.vdot(env.mu_0, np.einsum('i,ji', env.mu_0_major,
                                                   eval_curr_minor_reward(env, mu_disc, action_probs_minor,
                                                                          action_probs_major)[0][:, :,
                                                   mu_index_initial]))

            plot_vals.append(v_curr_1)
            plot_vals_2.append(v_curr_2)
            print(fr"{num_disc_mf}: {plot_vals}")

        plot_vals = np.array(plot_vals)
        plot_vals_2 = np.array(plot_vals_2)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(num_disc_mfs[::skip_n], plot_vals[::skip_n], linestyle, color=color,
                     label="$J^0$", alpha=0.5)
        subplot.scatter(num_disc_mfs[::skip_n], plot_vals[::skip_n], color=color,
                     label="$J^0$", alpha=0.5)
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot([min(num_disc_mfs), max(num_disc_mfs)], [plot_vals[-1]] * 2, linestyle, color=color,
                     label="$J^0$", alpha=0.5)
        subplot.scatter([min(num_disc_mfs), max(num_disc_mfs)], [plot_vals[-1]] * 2, color=color,
                     label="$J^0$", alpha=0.5)

        # linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_disc_mfs[::skip_n], plot_vals_2[::skip_n], linestyle, color=color,
        #              label="$J$", alpha=0.5)

        # if i==7:
        #     plt.legend()
        # plt.title(game)
        plt.grid('on')
        plt.xlabel(r'MF grid fineness $M$', fontsize=22)
        if i==5:
            plt.ylabel(r'$\hat J^0$', fontsize=22)
        plt.xlim([min(num_disc_mfs), max(num_disc_mfs)])
        # plt.ylim([0, 500])
        subplot.yaxis.set_major_locator(MaxNLocator(3, integer=True))

    """ Finalize plot """
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout(w_pad=0.0)
    plt.savefig(f'./figures/J_J0_discretization_maxent.pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.savefig(f'./figures/J_J0_discretization_maxent.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    plot()
