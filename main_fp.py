import itertools

import numpy as np

import args_parser
from envs.m3fg import MajorMinorMARLEnv
from utils import get_action_probs_from_Qs, get_new_action_probs_from_Qs, \
    find_best_response_major, \
    find_best_response_minor, eval_curr_major_reward, eval_curr_minor_reward, get_softmax_new_action_probs_from_Qs, \
    get_softmax_action_probs_from_Qs, find_discounted_best_response_minor, find_discounted_best_response_major, \
    eval_discounted_curr_major_reward, eval_discounted_curr_minor_reward

if __name__ == '__main__':
    config = args_parser.parse_config()
    env: MajorMinorMARLEnv = config['game'](**config)

    """ Epsilon net of simplex for mean field discretization """
    point_candidates = itertools.product(*[np.linspace(1 / config['num_disc_mf'] / 2, 1 - 1 / config['num_disc_mf'] / 2, config['num_disc_mf'])] * (env.minor_observation_space.n - 1))
    points = [p + (1 - sum(p),) for p in point_candidates if sum(p) <= 1]
    mu_disc = np.array(points).transpose()
    dists_to_net = np.sum(np.abs(np.expand_dims(env.mu_0, axis=-1) - mu_disc), axis=0)
    mu_index_initial = dists_to_net.argmin()

    major_Q_0 = [np.zeros((env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], env.major_action_space.n))]
    minor_Q_0 = [np.zeros((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n, mu_disc.shape[-1], env.minor_action_space.n))]

    if 'init_pi' in config and config['init_pi'] == 'first':
        major_Q_0 = [np.concatenate([
            np.ones((env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], 1)),
            np.zeros((env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], env.major_action_space.n - 1)),
        ], axis=-1)]
        minor_Q_0 = [np.concatenate([
            np.ones((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n, mu_disc.shape[-1], 1)),
            np.zeros((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n, mu_disc.shape[-1], env.minor_action_space.n - 1)),
        ], axis=-1)]
    elif 'init_pi' in config and config['init_pi'] == 'last':
        major_Q_0 = [np.concatenate([
            np.zeros((env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], env.major_action_space.n - 1)),
            np.ones((env.time_steps, env.major_observation_space.n, mu_disc.shape[-1], 1)),
        ], axis=-1)]
        minor_Q_0 = [np.concatenate([
            np.zeros((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n, mu_disc.shape[-1], env.minor_action_space.n - 1)),
            np.ones((env.time_steps, env.minor_observation_space.n, env.major_observation_space.n, mu_disc.shape[-1], 1)),
        ], axis=-1)]

    if config['softmax'] or ('init_pi' in config and config['init_pi'] == 'unif'):
        action_probs_minor = get_softmax_action_probs_from_Qs(np.array(minor_Q_0), temperature=config['temperature'])
        action_probs_major = get_softmax_action_probs_from_Qs(np.array(major_Q_0), temperature=config['temperature'])
    else:
        action_probs_minor = get_action_probs_from_Qs(np.array(minor_Q_0))
        action_probs_major = get_action_probs_from_Qs(np.array(major_Q_0))

    if config['variant'] == "omd":
        if config['inf']:
            y_major = 0 * find_discounted_best_response_major(env, mu_disc, action_probs_minor, gamma=config['gamma'])
            y_minor = 0 * find_discounted_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])
        else:
            y_major = 0 * find_best_response_major(env, mu_disc, action_probs_minor)
            y_minor = 0 * find_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major)

    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
            if config['variant'] == "omd":
                if config['inf']:
                    Q_pi_major = eval_discounted_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])[1]
                    Q_pi_minor = eval_discounted_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])[1]
                else:
                    Q_pi_major = eval_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major)[1]
                    Q_pi_minor = eval_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major)[1]

            if config['inf']:
                Q_br_major = find_discounted_best_response_major(env, mu_disc, action_probs_minor, gamma=config['gamma'])
                Q_br_minor = find_discounted_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])
            else:
                Q_br_major = find_best_response_major(env, mu_disc, action_probs_minor)
                Q_br_minor = find_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major)

            """ Evaluate current best response against current average policy """
            v_1 = np.vdot(env.mu_0_major, Q_br_major.max(axis=-1)[0, :, mu_index_initial])
            v_2 = np.vdot(env.mu_0, np.einsum('i,ji', env.mu_0_major, Q_br_minor.max(axis=-1)[0, :, :, mu_index_initial]))

            if config['inf']:
                v_curr_1 = np.vdot(env.mu_0_major, eval_discounted_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])[0][:, mu_index_initial])
                v_curr_2 = np.vdot(env.mu_0, np.einsum('i,ji', env.mu_0_major, eval_discounted_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=config['gamma'])[0][:, :, mu_index_initial]))
            else:
                v_curr_1 = np.vdot(env.mu_0_major, eval_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major)[0][:, mu_index_initial])
                v_curr_2 = np.vdot(env.mu_0, np.einsum('i,ji', env.mu_0_major, eval_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major)[0][:, :, mu_index_initial]))

            print(f"{config['exp_dir']} {iteration}: expl major: {v_1 - v_curr_1} expl minor: {v_2 - v_curr_2}, ... major br achieves {v_1} vs. {v_curr_1}, minor br achieves {v_2} vs {v_curr_2}", flush=True)
            fo.write(f"{config['exp_dir']} {iteration}: expl major: {v_1 - v_curr_1} expl minor: {v_2 - v_curr_2}, ... major br achieves {v_1} vs. {v_curr_1}, minor br achieves {v_2} vs {v_curr_2}")
            fo.write('\n')

            if config['variant'] == "fpi":
                if config['softmax']:
                    action_probs_minor = get_softmax_action_probs_from_Qs(np.array([Q_br_minor]), temperature=config['temperature'])
                    action_probs_major = get_softmax_action_probs_from_Qs(np.array([Q_br_major]), temperature=config['temperature'])
                else:
                    action_probs_minor = get_action_probs_from_Qs(np.array([Q_br_minor]))
                    action_probs_major = get_action_probs_from_Qs(np.array([Q_br_major]))
            elif config['variant'] == "fp":
                if config['softmax']:
                    action_probs_minor = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs_minor, np.array([Q_br_minor]), temperature=config['temperature'])
                    action_probs_major = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs_major, np.array([Q_br_major]), temperature=config['temperature'])
                else:
                    action_probs_minor = get_new_action_probs_from_Qs(iteration + 1, action_probs_minor, np.array([Q_br_minor]))
                    action_probs_major = get_new_action_probs_from_Qs(iteration + 1, action_probs_major, np.array([Q_br_major]))
            elif config['variant'] == "sfp":
                if iteration % (config['inner_per_outer_iterations'] + 1) == 0:
                    if config['softmax']:
                        action_probs_major = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs_major, np.array([Q_br_major]), temperature=config['temperature'])
                    else:
                        action_probs_major = get_new_action_probs_from_Qs(iteration + 1, action_probs_major, np.array([Q_br_major]))
                else:
                    if config['softmax']:
                        action_probs_minor = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs_minor, np.array([Q_br_minor]), temperature=config['temperature'])
                    else:
                        action_probs_minor = get_new_action_probs_from_Qs(iteration + 1, action_probs_minor, np.array([Q_br_minor]))
            elif config['variant'] == "omd":
                y_minor += Q_pi_minor
                y_major += Q_pi_major
                action_probs_minor = get_softmax_action_probs_from_Qs(np.array([y_minor]), temperature=config['temperature'])
                action_probs_major = get_softmax_action_probs_from_Qs(np.array([y_major]), temperature=config['temperature'])

            np.save(config['exp_dir'] + f"action_probs_major.npy", action_probs_major)
            np.save(config['exp_dir'] + f"action_probs_minor.npy", action_probs_minor)
            np.save(config['exp_dir'] + f"major_best_response.npy", Q_br_major)
            np.save(config['exp_dir'] + f"minor_best_response.npy", Q_br_minor)

    # np.save(config['exp_dir'] + f"action_probs_major.npy", action_probs_major)
    # np.save(config['exp_dir'] + f"action_probs_minor.npy", action_probs_minor)

    """ Plot a trajectory """
    # plot_debug_trajectory(env, mu_disc, action_probs_minor, action_probs_major, inf_discounted=config['inf'])
