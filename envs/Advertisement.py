import itertools

import numpy as np
from gym.spaces import Discrete

from envs.m3fg import MajorMinorMARLEnv


class Advertisement(MajorMinorMARLEnv):
    """
    Models the Advertisement problem.
    """

    def __init__(self, change_rate: float = 0.3, major_change_rate: float = 0.05, time_steps: int = 100,
                 c_c: float = 0.75, c_o: float = 1.0, c_ad: float = 1.0, c_crowd: float = 1.0,
                 c0_ad: float = 0.1, c0_monopoly: float = 1,
                 k_0 = 0.2, k_y = 0.5, k_u0 = 0.7, k_u = 1.0,
                 num_agents: int = 300, **kwargs):
        self.change_rate = change_rate
        self.major_change_rate = major_change_rate
        self.c_c = c_c
        self.c_o = c_o
        self.c_ad = c_ad
        self.c_crowd = c_crowd
        self.c0_ad = c0_ad
        self.c0_monopoly = c0_monopoly
        self.k_0 = k_0
        self.k_y = k_y
        self.k_u0 = k_u0
        self.k_u = k_u

        minor_observation_space = Discrete(2)  # c1, c2 aggressive
        minor_action_space = Discrete(2)  # c, o
        major_observation_space = Discrete(2)  # c1, c2
        major_action_space = Discrete(3)  # n, c1, c2

        mu_0 = np.array([0.5, 0.5, ])
        mu_0_major = np.array([1, 0, ])

        super().__init__(minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents, mu_0, mu_0_major, **kwargs)

    def a1(self, y, u_major):
        return self.k_0 + self.k_y * (y==0) + self.k_u0 * (u_major==1)

    def a2(self, y, u_major):
        return self.k_0 + self.k_y * (y==1) + self.k_u0 * (u_major==2)

    def next_states(self, t, xs, y, us, u_major):
        a1 = self.a1(y, u_major)
        a2 = self.a2(y, u_major)

        changes_plus = np.random.rand(self.num_agents) < self.change_rate * a2 / a1 * (0.2 + self.k_u * us)
        changes_minus = np.random.rand(self.num_agents) < self.change_rate * a1 / a2 * (0.2 + self.k_u * us)
        new_xs = (xs == 0) * changes_plus \
                 + (xs == 1) * (1 - changes_minus)

        new_y = (1-y) if np.random.rand() < self.major_change_rate else y

        return new_xs, new_y

    def get_P(self, mu_disc):
        P = np.zeros((self.minor_action_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_observation_space.n, self.minor_observation_space.n))

        for y, u_major, u in itertools.product(range(2), range(3), range(2)):
            a1 = self.a1(y, u_major)
            a2 = self.a2(y, u_major)

            P[u, y, u_major, ..., 0, 1] += np.minimum(1, self.change_rate * a2 / a1 * (0.2 + self.k_u * u))
            P[u, y, u_major, ..., 0, 0] += 1 - P[u, y, u_major, ..., 0, 1]
            P[u, y, u_major, ..., 1, 0] += np.minimum(1, self.change_rate * a1 / a2 * (0.2 + self.k_u * u))
            P[u, y, u_major, ..., 1, 1] += 1 - P[u, y, u_major, ..., 1, 0]

        return P

    def get_P_0(self, mu_disc):
        P_major = np.zeros((self.major_observation_space.n, self.major_action_space.n, mu_disc.shape[-1],
                            self.major_observation_space.n))

        P_major[0, ..., 1] = self.major_change_rate
        P_major[0, ..., 0] = 1 - P_major[0, ..., 1]
        P_major[1, ..., 0] = self.major_change_rate
        P_major[1, ..., 1] = 1 - P_major[1, ..., 0]

        return P_major

    def reward(self, t, xs, y, us, u_major):
        mu_1 = np.mean(xs == 0)
        mu_2 = np.mean(xs == 1)

        a1 = self.a1(y, u_major)
        a2 = self.a2(y, u_major)

        minor_rewards = self.c_crowd * ((mu_1 - mu_2) * (xs == 0) + (mu_2 - mu_1) * (xs == 1)) \
                        + self.c_ad * (a1 * (xs == 0) + a2 * (xs == 1)) \
                        - self.c_c * (us == 0) - self.c_o * (us == 1)
        major_reward = - self.c0_monopoly * np.abs(mu_1 - mu_2) - self.c0_ad * ((u_major==1) + (u_major==2))
        return minor_rewards, major_reward

    def get_R(self, mu_disc):
        R = np.zeros((self.minor_observation_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_action_space.n))

        R[0, ...] += self.c_crowd * np.expand_dims(mu_disc[0] - mu_disc[1], (0, 2))
        R[1, ...] += self.c_crowd * np.expand_dims(mu_disc[1] - mu_disc[0], (0, 2))

        for x, y, u_major in itertools.product(range(2), range(2), range(3)):
            a1 = self.a1(y, u_major)
            a2 = self.a2(y, u_major)

            R[x, y, u_major] += self.c_ad * (a1 * (x == 0) + a2 * (x == 1))

        R[..., 0] -= self.c_c
        R[..., 1] -= self.c_o

        return R

    def get_R_0(self, mu_disc):
        R_0 = np.zeros((self.major_observation_space.n, mu_disc.shape[-1], self.major_action_space.n))

        R_0 -= self.c0_monopoly * np.expand_dims(np.abs(mu_disc[1] - mu_disc[0]), (0, 2))

        R_0[..., 1] -= self.c0_ad
        R_0[..., 2] -= self.c0_ad

        return R_0
