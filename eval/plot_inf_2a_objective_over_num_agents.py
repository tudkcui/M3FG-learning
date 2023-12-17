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

    games = ['SIS', 'Buffet', 'Advertisement',]
    variants = ["fp", ] * len(games)
    num_disc_mfs = [120, ] * len(games) # list(range(2, 30, 2)) + list(range(30, 210, 10))
    num_trials = 1000
    rerun = False
    stationary = True
    eval_timesteps = 500

    for game, num_disc_mf, variant in zip(games, num_disc_mfs, variants):

        num_agentss = [2, 5, 10, 20, 50, 100, 200] if game != "Buffet" else [2, 4, 6, 8, 10, 20, 40]
        num_trials = 1000 if game != "Buffet" else 5000
        # rerun = (game == "Buffet")

        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(2, len(games), i)
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
                         (1, 0),
                         xytext=(-36, +32 if i != 1 and i != 3 else 110),
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

        for num_agents in num_agentss:
            try:
                config = args_parser.generate_config_from_kw(game=game, variant=variant,
                                                             fp_iterations=100 if variant == 'fpi' else 1000,
                                                             num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary)

                action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
                action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
                major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
                minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")
                files = find('stdout', config['exp_dir'])
                with open(max(files, key=os.path.getctime), 'r') as fi:
                    fi_lines = fi.readlines()
                    line = fi_lines[-1]
                    fields = line.split(" ")
                    major_V_disc = float(fields[14][:-1])
                    minor_V_disc = float(fields[20][:-1])

                save_dir = config['exp_dir']

                if not rerun and os.path.exists(config['exp_dir'] + f"val0s_{num_agents}_{num_trials}.npy"):
                    val0s = np.load(config['exp_dir'] + f"val0s_{num_agents}_{num_trials}.npy")
                    vals = np.load(config['exp_dir'] + f"vals_{num_agents}_{num_trials}.npy")
                else:
                    config = args_parser.generate_config_from_kw(game=game, variant=variant,
                                                                 fp_iterations=100 if variant == 'fpi' else 1000,
                                                                 num_agents=num_agents, num_disc_mf=num_disc_mf,
                                                                 inf=stationary)
                    env = config['game'](**config)
                    point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2,
                                                                       1 - 1 / config['num_disc_mf'] / 2,
                                                                       config['num_disc_mf'])] * (
                                                                      env.minor_observation_space.n - 1))
                    points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
                    mu_disc = np.array(points).transpose()

                    vals = []
                    val0s = []
                    for _ in range(num_trials):
                        """ Plot mean field of states (for debugging) """
                        xs, y = env.reset()
                        mf_state = [np.mean(xs == i) for i in range(env.minor_observation_space.n)]
                        mf_states = [mf_state]
                        xss = [xs]
                        uss = []
                        ys = [y]
                        u0s = []
                        indices = []
                        val = 0
                        val0 = 0
                        gamma = 0.99
                        curr_gammas = 1.0
                        for t in range(eval_timesteps):
                            r""" Project current mf to epsilon net in L1 dist """
                            dists_to_net = np.sum(np.abs(np.expand_dims(mf_state, axis=-1) - mu_disc), axis=0)
                            mu_closest_index = dists_to_net.argmin(-1)

                            # action_minors = np.random.rand() < action_probs_minor[0 if stationary else t, :, y, mu_closest_index, 1][xs]
                            action_major = np.random.choice(range(env.major_action_space.n), p=action_probs_major[
                                0 if stationary else t, y, mu_closest_index])

                            """ TEST """
                            cum_ps = np.cumsum(action_probs_minor[0 if stationary else t, :, y, mu_closest_index, ][xs],
                                               axis=-1)
                            action_minors = np.zeros((num_agents,), dtype=int)
                            uniform_samples = np.random.uniform(0, 1, size=num_agents)
                            for idx in range(env.minor_observation_space.n):
                                action_minors += idx * np.logical_and(
                                    uniform_samples >= (cum_ps[:, idx - 1] if idx - 1 >= 0 else 0.0),
                                    uniform_samples < cum_ps[:, idx])

                            (xs, y), rewards, done, info = env.step(action_minors, action_major)
                            val += curr_gammas * rewards[0]
                            val0 += curr_gammas * rewards[1]

                            curr_gammas *= gamma

                            xss.append(xs)
                            uss.append(action_minors)
                            ys.append(y)
                            u0s.append(action_major)
                            indices.append(mu_closest_index)

                            mf_state = [np.mean(xs == i) for i in range(env.minor_observation_space.n)]
                            mf_states.append(mf_state)

                        # print(f"Values of major: {val0} minor avg: {np.mean(val)}")
                        val0s.append(val0)
                        vals.append(val[0])

                    val0s = np.array(val0s)
                    vals = np.array(vals)
                    np.save(save_dir + f"val0s_{num_agents}_{num_trials}.npy", val0s)
                    np.save(save_dir + f"vals_{num_agents}_{num_trials}.npy", vals)

                print(f"Values of major: {np.mean(val0s)} +- {2 * np.std(val0s) / np.sqrt(len(val0s))} "
                      f"minor avg:  {np.mean(vals)} +- {2 * np.std(vals) / np.sqrt(len(vals))}")
                plot_vals.append(np.mean(vals))
                plot_vals_2.append(2 * np.std(vals) / np.sqrt(len(vals)))

            except Exception as e:
                print(f"FAILED TO MAKE SUBPLOT: {e}")

        plot_vals = np.array(plot_vals)
        plot_vals_2 = np.array(plot_vals_2)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n] + plot_vals_2[::skip_n], linestyle, color=color,
        #              label='_nolabel_', alpha=0.5)
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n] - plot_vals_2[::skip_n], linestyle, color=color,
        #              label='_nolabel_', alpha=0.5)
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n], linestyle, color=color,
        #              label="$J^0$", alpha=0.85)
        # subplot.fill_between(num_agentss[::skip_n], plot_vals[::skip_n] - plot_vals_2[::skip_n],
        #                      plot_vals[::skip_n] + plot_vals_2[::skip_n], color=color, alpha=0.15)

        subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n], linestyle, color=color, label="$J^0$")
        subplot.scatter(num_agentss[::skip_n], plot_vals[::skip_n], color=color, label="__nolabel__")
        subplot.errorbar(num_agentss[::skip_n], plot_vals[::skip_n], yerr=plot_vals_2[::skip_n], color=color,
                         label="__nolabel__", alpha=0.85)

        # linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_agentss[::skip_n], plot_vals_2[::skip_n], linestyle, color=color,
        #              label="$J$", alpha=0.5)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot([num_agentss[0], num_agentss[-1]],
                     [minor_V_disc] * 2, linestyle, color=color,
                     label='_nolabel_', alpha=0.5)

        if i == 7:
            plt.legend()
        # plt.title(game + " " + variant)
        plt.grid('on')
        # plt.xlabel(r'Num agents', fontsize=22)
        if i == 2:
            plt.ylabel(r'$\hat J$', fontsize=22)
        plt.xlim([min(num_agentss), max(num_agentss)])
        # plt.ylim([0, 500])

    rerun = False
    for game, num_disc_mf, variant in zip(games, num_disc_mfs, variants):

        num_agentss = [2, 5, 10, 20, 50, 100, 200] if game != "Buffet" else [2, 4, 6, 8, 10, 20, 40]
        num_trials = 1000 if game != "Buffet" else 5000

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

        for num_agents in num_agentss:
            config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                         num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary)

            action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
            action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
            major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
            minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")
            files = find('stdout', config['exp_dir'])
            with open(max(files, key=os.path.getctime), 'r') as fi:
                fi_lines = fi.readlines()
                line = fi_lines[-1]
                fields = line.split(" ")
                major_V_disc = float(fields[14][:-1])
                minor_V_disc = float(fields[20][:-1])

            save_dir = config['exp_dir']

            # if game == 'Advertisement':
            #     config = args_parser.generate_config_from_kw(game=game, variant='fpi', fp_iterations=20,
            #                                                  num_agents=1000, num_disc_mf=100, inf=stationary,
            #                                                  k_0=0.1, k_y=0.4, k_u0=0.6, k_u=0.9)
            #
            #     action_probs_major = np.load(config['exp_dir'] + f"action_probs_major.npy")
            #     action_probs_minor = np.load(config['exp_dir'] + f"action_probs_minor.npy")
            #     major_best_response = np.load(config['exp_dir'] + f"major_best_response.npy")
            #     minor_best_response = np.load(config['exp_dir'] + f"minor_best_response.npy")
            #     files = find('stdout', config['exp_dir'])
            #     with open(max(files, key=os.path.getctime), 'r') as fi:
            #         fi_lines = fi.readlines()
            #         line = fi_lines[-1]
            #         fields = line.split(" ")
            #         major_V_disc = float(fields[14][:-1])
            #         minor_V_disc = float(fields[20][:-1])
            #
            #     save_dir = config['exp_dir']
            #     num_trials = 20

            if not rerun and os.path.exists(save_dir + f"val0s_{num_agents}_{num_trials}.npy"):
                val0s = np.load(save_dir + f"val0s_{num_agents}_{num_trials}.npy")
                vals = np.load(save_dir + f"vals_{num_agents}_{num_trials}.npy")
            else:
                raise NotImplementedError

            print(f"Values of major: {np.mean(val0s)} +- {2 * np.std(val0s) / np.sqrt(len(val0s))} "
                  f"minor avg:  {np.mean(vals)} +- {2 * np.std(vals) / np.sqrt(len(vals))}")
            plot_vals.append(np.mean(val0s))
            plot_vals_2.append(2 * np.std(val0s) / np.sqrt(len(val0s)))

        plot_vals = np.array(plot_vals)
        plot_vals_2 = np.array(plot_vals_2)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n] + plot_vals_2[::skip_n], linestyle, color=color,
        #              label='_nolabel_', alpha=0.5)
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n] - plot_vals_2[::skip_n], linestyle, color=color,
        #              label='_nolabel_', alpha=0.5)
        # subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n], linestyle, color=color,
        #              label="$J^0$", alpha=0.85)
        # subplot.fill_between(num_agentss[::skip_n], plot_vals[::skip_n] - plot_vals_2[::skip_n],
        #                      plot_vals[::skip_n] + plot_vals_2[::skip_n], color=color, alpha=0.15)

        subplot.plot(num_agentss[::skip_n], plot_vals[::skip_n], linestyle, color=color, label="$J^0$")
        subplot.scatter(num_agentss[::skip_n], plot_vals[::skip_n], color=color, label="__nolabel__")
        subplot.errorbar(num_agentss[::skip_n], plot_vals[::skip_n], yerr=plot_vals_2[::skip_n], color=color,
                         label="__nolabel__", alpha=0.85, capsize=3)

        # linestyle = linestyle_cycler.__next__()['linestyle']
        # subplot.plot(num_agentss[::skip_n], plot_vals_2[::skip_n], linestyle, color=color,
        #              label="$J$", alpha=0.5)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot([num_agentss[0], num_agentss[-1]],
                     [major_V_disc] * 2, linestyle, color=color,
                     label='_nolabel_',)

        # if i==7:
        #     plt.legend()
        # plt.title(game + " " + variant)
        plt.grid('on')
        plt.xlabel(r'Num agents', fontsize=22)
        if i == 5:
            plt.ylabel(r'$\hat J^0$', fontsize=22)
        plt.xlim([min(num_agentss), max(num_agentss)])
        # plt.ylim([0, 500])

    """ Finalize plot """
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout(w_pad=0.0)
    plt.savefig(f'./figures/inf_J_J0_num_agents.pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.savefig(f'./figures/inf_J_J0_num_agents.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    plot()
