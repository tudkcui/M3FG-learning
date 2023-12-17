import itertools

import numpy as np
from matplotlib import pyplot as plt, cycler

from envs.Advertisement import Advertisement
from envs.Buffet import Buffet
from envs.SIS import SIS


def plot_debug_trajectory(env, mu_disc, action_probs_minor, action_probs_major, inf_discounted=False):
    vals = []
    val0s = []
    for _ in range(1):
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
        for t in range(env.time_steps):
            r""" Project current mf to epsilon net in L1 dist """
            dists_to_net = np.sum(np.abs(np.expand_dims(mf_state, axis=-1) - mu_disc), axis=0)
            mu_closest_index = dists_to_net.argmin(-1)

            action_minors = np.array([
                np.random.choice(range(env.minor_action_space.n), p=action_probs_minor[0 if inf_discounted else t, xs[i], y, mu_closest_index])
                for i in range(env.num_agents)
            ])
            action_major = np.random.choice(range(env.major_action_space.n), p=action_probs_major[0 if inf_discounted else t, y, mu_closest_index])

            """ TEST """
            # action_major = 1

            (xs, y), rewards, done, info = env.step(action_minors, action_major)
            val += rewards[0]
            val0 += rewards[1]

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

    print(f"Empirical return of a trajectory major: {np.mean(val0s)} +- {2 * np.std(val0s) / np.sqrt(len(val0s))} "
          f"minor avg:  {np.mean(vals)} +- {2 * np.std(vals) / np.sqrt(len(vals))}")

    plt.subplot(2, 1, 1)
    clist = itertools.cycle(cycler(color='rbkgcmy'))
    linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))

    """ Plot MF """
    # plt.subplot(5, 1, 1)
    # plt.ylabel("MF")
    # plt.ylim([0, 1])
    mf_states = np.array(mf_states)
    for i in range(1, env.minor_observation_space.n):
        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        plt.plot(range(env.time_steps + 1), mf_states[:, i], linestyle, color=color, label=fr'$\mu_t({"I" if isinstance(env, SIS) else i})$', linewidth=2)

    """ Plot mf index """
    # plt.subplot(5, 1, 3)
    # plt.ylabel("MF bin")
    # plt.plot(range(env.time_steps), indices, color=color, label='_nolabel_', linewidth=2)

    """ Plot y """
    # plt.subplot(5, 1, 4)
    # plt.ylabel("$x^0$")
    if isinstance(env, SIS):
        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        plt.plot(range(env.time_steps + 1), ys, linestyle, color=color, label="$x^0$", linewidth=2)
    elif isinstance(env, Buffet):
        for i in range(env.minor_observation_space.n):
            color = clist.__next__()['color']
            linestyle = linestyle_cycler.__next__()['linestyle']
            plt.plot(range(env.time_steps + 1), env.to_fillings(np.array(ys))[:, i] / 5, linestyle, color=color, label=f'$x^0_{i} / 5$', linewidth=2)
        plt.plot([0, env.time_steps], [0, 0], alpha=0)
        plt.plot([0, env.time_steps], [1, 1], alpha=0)
    elif isinstance(env, Advertisement):
        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        plt.plot(range(env.time_steps + 1), ys, linestyle, color=color, label="$x^0$", linewidth=2)

    lgd1 = plt.legend(loc="lower right", bbox_to_anchor=(1.32 if isinstance(env, Buffet) else 1.28,
                                                         -0.5 if isinstance(env, Buffet) else 0))
    plt.grid()
    plt.xlim([0, env.time_steps])

    plt.subplot(2, 1, 2)
    clist = itertools.cycle(cycler(color='rbkgcmy'))
    linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
    """ Plot mean of minor actions """
    # plt.subplot(5, 1, 2)
    # plt.ylabel("mean us")
    color = clist.__next__()['color']
    linestyle = linestyle_cycler.__next__()['linestyle']
    plt.plot(range(env.time_steps), np.mean(uss, axis=1), linestyle, color=color, label=r'$\frac{1}{N} \sum_i u_t^i$', linewidth=2)

    """ Plot u0 """
    # plt.subplot(5, 1, 5)
    # plt.ylabel("$u^0$")
    color = clist.__next__()['color']
    linestyle = linestyle_cycler.__next__()['linestyle']
    if isinstance(env, Advertisement):
        plt.plot(range(env.time_steps), np.array(u0s)/2, linestyle, color=color, label="$u^0/2$", linewidth=2)
    else:
        plt.plot(range(env.time_steps), u0s, linestyle, color=color, label="$u^0$", linewidth=2)

    lgd2 = plt.legend(loc="lower right", bbox_to_anchor=(1.32, 0))
    plt.grid()
    plt.xlim([0, env.time_steps])
    plt.xlabel("$t$")

    """ Finalize plot """
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    from pathlib import Path
    Path(f"./figures/").mkdir(parents=True, exist_ok=True)
    if inf_discounted:
        plt.savefig(f'./figures/inf_out_{env.__class__.__name__}.pdf', bbox_extra_artists=(lgd1, lgd2), bbox_inches='tight', transparent=True, pad_inches=0.1)
        plt.savefig(f'./figures/inf_out_{env.__class__.__name__}.png', bbox_extra_artists=(lgd1, lgd2,), bbox_inches='tight', transparent=True, pad_inches=0.1)
    else:
        plt.savefig(f'./figures/out_{env.__class__.__name__}.pdf', bbox_extra_artists=(lgd1, lgd2,), bbox_inches='tight', transparent=True, pad_inches=0.1)
        plt.savefig(f'./figures/out_{env.__class__.__name__}.png', bbox_extra_artists=(lgd1, lgd2,), bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


def get_action_probs_from_Qs(Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    return b.reshape(Qs.shape).mean(0)


def get_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    new_probs = b.reshape(Qs.shape).mean(0)
    return (old_probs * num_averages_yet + new_probs) / (num_averages_yet + 1)


def find_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major):
    Qs = []
    V_t_next = np.zeros((env.minor_observation_space.n, env.major_observation_space.n, len(mu_disc[0])))
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_transition_probs_minor(mu_disc, action_probs_minor[t], action_probs_major[t])
        Q_t = env.get_reward_vec_minor(mu_disc, action_probs_minor[t], action_probs_major[t]) \
              + np.einsum('ijklmno,mno->ijkl', P_t, V_t_next)
        V_t_next = np.max(Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return out_Qs


def find_best_response_major(env, mu_disc, action_probs_minor):
    Qs = []
    V_t_next = np.zeros((env.major_observation_space.n, len(mu_disc[0])))
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_transition_probs_major(mu_disc, action_probs_minor[t])
        Q_t = env.get_reward_vec_major(mu_disc, action_probs_minor[t]) \
              + np.einsum('ijklm,lm', P_t, V_t_next)
        V_t_next = np.max(Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return out_Qs


def eval_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major):
    Qs = []
    V_t_next = np.zeros((env.minor_observation_space.n, env.major_observation_space.n, len(mu_disc[0])))
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_transition_probs_minor(mu_disc, action_probs_minor[t], action_probs_major[t])
        Q_t = env.get_reward_vec_minor(mu_disc, action_probs_minor[t], action_probs_major[t]) \
              + np.einsum('ijklmno,mno->ijkl', P_t, V_t_next)
        V_t_next = np.sum(action_probs_minor[t] * Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return V_t_next, out_Qs


def eval_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major):
    Qs = []
    V_t_next = np.zeros((env.major_observation_space.n, len(mu_disc[0])))
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_transition_probs_major(mu_disc, action_probs_minor[t])
        Q_t = env.get_reward_vec_major(mu_disc, action_probs_minor[t]) \
              + np.einsum('ijklm,lm', P_t, V_t_next)
        V_t_next = np.sum(action_probs_major[t] * Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return V_t_next, out_Qs


def find_discounted_best_response_minor(env, mu_disc, action_probs_minor, action_probs_major, gamma=0.99):
    V = np.zeros((env.minor_observation_space.n, env.major_observation_space.n, len(mu_disc[0])))
    P = env.get_transition_probs_minor(mu_disc, action_probs_minor[0], action_probs_major[0])

    eps = 999999
    while eps > 1e-5:
        Q = env.get_reward_vec_minor(mu_disc, action_probs_minor[0], action_probs_major[0]) \
              + gamma * np.einsum('ijklmno,mno->ijkl', P, V)
        V_new = np.max(Q, axis=-1)

        eps = np.sum(np.abs(V - V_new))
        V = V_new
    return np.array([Q])


def find_discounted_best_response_major(env, mu_disc, action_probs_minor, gamma=0.99):
    V = np.zeros((env.major_observation_space.n, len(mu_disc[0])))
    P = env.get_transition_probs_major(mu_disc, action_probs_minor[0])

    eps = 999999
    while eps > 1e-5:
        Q = env.get_reward_vec_major(mu_disc, action_probs_minor[0]) \
              + gamma * np.einsum('ijklm,lm', P, V)
        V_new = np.max(Q, axis=-1)

        eps = np.sum(np.abs(V - V_new))
        V = V_new
    return np.array([Q])


def eval_discounted_curr_minor_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=0.99):
    V = np.zeros((env.minor_observation_space.n, env.major_observation_space.n, len(mu_disc[0])))
    P = env.get_transition_probs_minor(mu_disc, action_probs_minor[0], action_probs_major[0])

    eps = 999999
    while eps > 1e-5:
        Q = env.get_reward_vec_minor(mu_disc, action_probs_minor[0], action_probs_major[0]) \
              + gamma * np.einsum('ijklmno,mno->ijkl', P, V)
        V_new = np.sum(action_probs_minor[0] * Q, axis=-1)

        eps = np.sum(np.abs(V - V_new))
        V = V_new
    return V, np.array([Q])


def eval_discounted_curr_major_reward(env, mu_disc, action_probs_minor, action_probs_major, gamma=0.99):
    V = np.zeros((env.major_observation_space.n, len(mu_disc[0])))
    P = env.get_transition_probs_major(mu_disc, action_probs_minor[0])

    eps = 999999
    while eps > 1e-5:
        Q = env.get_reward_vec_major(mu_disc, action_probs_minor[0]) \
              + gamma * np.einsum('ijklm,lm', P, V)
        V_new = np.sum(action_probs_major[0] * Q, axis=-1)

        eps = np.sum(np.abs(V - V_new))
        V = V_new
    return V, np.array([Q])


def get_softmax_action_probs_from_Qs(Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    return b.reshape(Qs.shape).mean(0)


def get_softmax_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    new_probs = b.reshape(Qs.shape).mean(0)
    return (old_probs * num_averages_yet + new_probs) / (num_averages_yet + 1)
