import fnmatch
import itertools
import os
import string

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

import args_parser
from utils import plot_debug_trajectory


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

    games = ['Buffet',]
    variants = ["fp", ]
    num_disc_mfs = [120, ]
    num_agentss = [300]  # list(range(2, 30, 2)) + list(range(30, 210, 10))
    stationary = False

    for game, num_disc_mf, variant in zip(games, num_disc_mfs, variants):

        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        subplot = plt.subplot(1, 2, i)
        subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        plot_vals = []
        plot_vals_2 = []

        for num_agents in num_agentss:
            config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                         num_agents=1000, num_disc_mf=num_disc_mf, inf=stationary, temperature=0.1, softmax=0)

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
            config = args_parser.generate_config_from_kw(game=game, variant=variant, fp_iterations=100 if variant == 'fpi' else 1000,
                                                         num_agents=num_agents, num_disc_mf=num_disc_mf, inf=stationary, temperature=0.1, softmax=0)
            env = config['game'](**config)
            point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2, 1 - 1 / config['num_disc_mf'] / 2, config['num_disc_mf'])] * (env.minor_observation_space.n - 1))
            points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
            mu_disc = np.array(points).transpose()

            plot_debug_trajectory(env, mu_disc, action_probs_minor, action_probs_major, inf_discounted=stationary)


if __name__ == '__main__':
    plot()
