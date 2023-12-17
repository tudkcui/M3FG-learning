import argparse

from envs.Advertisement import Advertisement
from envs.Buffet import Buffet
from envs.Cyber import Cyber
from envs.RegulatedDuopoly import RegulatedDuopoly
from envs.SIS import SIS


def parse_args():
    parser = argparse.ArgumentParser(description="M3FG")
    parser.add_argument('--cores', type=int, help='number of cores per worker', default=1)
    parser.add_argument('--game', help='game to solve', default="SIS")
    parser.add_argument('--iterations', type=int, help='number of training iterations', default=100)
    parser.add_argument('--fp_iterations', type=int, help='number of fp iterations', default=7)
    parser.add_argument('--num_disc_mf', type=int, help='number of mf discretizations', default=100)
    parser.add_argument('--inner_per_outer_iterations', type=int, help='stackelberg inner iterations', default=5)
    parser.add_argument('--id', type=int, help='experiment id', default=0)
    parser.add_argument('--num_agents', type=int, help='Number of agents simulated', default=1000)

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    parser.add_argument("--softmax", action="store_true", default=False, help="Use softmax instead of argmax")
    parser.add_argument("--inf", action="store_true", default=False, help="infinite horizon")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature")
    parser.add_argument("--variant", default="fpi", choices=["fpi", "fp", "sfp", "omd"])

    parser.add_argument("--init_pi", choices=["first", "last", "unif"])

    parsed, unknown = parser.parse_known_args()
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    def isint(num):
        try:
            int(num)
            return True
        except ValueError:
            return False
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0],
                                type=int if isint(arg.split('=')[1]) else float if isfloat(arg.split('=')[1]) else str)

    return parser.parse_args()


def generate_config(args):
    return generate_config_from_kw(**vars(args))


def generate_config_from_kw(temperature=0.1, softmax=0, init_pi=None, **kwargs):
    kwargs['temperature'], kwargs['softmax'], kwargs['init_pi'] = temperature, softmax, init_pi

    kwargs['exp_dir'] = "./results/%s_%s_%d_%d_%d_%d_%f_%d" \
               % (kwargs['game'], kwargs['variant'], kwargs['fp_iterations'], kwargs['num_agents'],
                  kwargs['num_disc_mf'], kwargs['inf'], kwargs['temperature'], kwargs['softmax'], )
    if 'max_fill_state' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['max_fill_state']}"
    if 'k_0' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['k_0']}"
    if 'k_y' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['k_y']}"
    if 'k_u0' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['k_u0']}"
    if 'k_u' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['k_u']}"
    if 'c_c' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c_c']}"
    if 'c_o' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c_o']}"
    if 'c_ad' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c_ad']}"
    if 'c_crowd' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c_crowd']}"
    if 'c0_ad' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c0_ad']}"
    if 'c0_monopoly' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['c0_monopoly']}"
    if 'init_pi' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['init_pi']}"

    kwargs['exp_dir'] += f"/"

    from pathlib import Path
    Path(f"{kwargs['exp_dir']}").mkdir(parents=True, exist_ok=True)

    if kwargs['game'] == 'SIS':
        kwargs['game'] = SIS
    elif kwargs['game'] == 'Cyber':
        kwargs['game'] = Cyber
    elif kwargs['game'] == 'Buffet':
        kwargs['game'] = Buffet
    elif kwargs['game'] == 'RegulatedDuopoly':
        kwargs['game'] = RegulatedDuopoly
    elif kwargs['game'] == 'Advertisement':
        kwargs['game'] = Advertisement
    else:
        raise NotImplementedError

    return kwargs


def parse_config():
    args = parse_args()
    return generate_config(args)
