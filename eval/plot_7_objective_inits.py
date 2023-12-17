import fnmatch
import itertools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import MaxNLocator

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

    games = ['SIS', 'Buffet', 'Advertisement']
    variants = ["fp",] * len(games)
    num_disc_mfs = range(60, 130, 10)
    init_pis = ["first", "last", "unif"]
    stationary = False

    for game, variant in zip(games, variants):

        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(2, len(games), i)
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
                         (1, 1),
                         xytext=(-36, -10 if i != 3 else -95),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        for init_pi in init_pis:
            plot_vals = []
            plot_vals_2 = []

            for num_disc_mf in num_disc_mfs:
                try:
                    config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                                 num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary,
                                                                 init_pi=init_pi)
                    files = find('stdout', config['exp_dir'])

                    with open(max(files, key=os.path.getctime), 'r') as fi:
                        fi_lines = fi.readlines()
                        for line in fi_lines[-1:]:
                            fields = line.split(" ")
                            plot_vals.append(float(fields[14][:-1]))
                            plot_vals_2.append(float(fields[20][:-1]))

                except Exception as e:
                    print(f"FAILED TO MAKE SUBPLOT: {e}")

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

        for init_pi in init_pis:
            plot_vals = []
            plot_vals_2 = []

            for num_disc_mf in num_disc_mfs:
                try:
                    config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                                 num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary,
                                                                 init_pi=init_pi)
                    files = find('stdout', config['exp_dir'])

                    with open(max(files, key=os.path.getctime), 'r') as fi:
                        fi_lines = fi.readlines()
                        for line in fi_lines[-1:]:
                            fields = line.split(" ")
                            plot_vals.append(float(fields[14][:-1]))
                            plot_vals_2.append(float(fields[20][:-1]))

                except Exception as e:
                    print(f"FAILED TO MAKE SUBPLOT: {e}")

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
    plt.savefig(f'./figures/J_J0_discretization_init_pis.pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.savefig(f'./figures/J_J0_discretization_init_pis.png', bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    plot()
