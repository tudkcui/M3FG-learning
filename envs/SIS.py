import numpy as np
from gym.spaces import Discrete

from envs.m3fg import MajorMinorMARLEnv


class SIS(MajorMinorMARLEnv):
    """
    Models the SIS game.
    """

    def __init__(self, infection_rate: float = 0.8, recovery_rate: float = 0.2, time_steps: int = 150,
                 initial_infection_prob = 0.2, initial_modifier_prob = 0.5, modifier_jump_prob: float = 0.4,
                 c_I: float = 0.75, c_P: float = 0.5, delta_t: float = 0.25,
                 c0_I: float = 2, c0_F: float = 1, num_agents: int = 300, **kwargs):
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infection_prob = initial_infection_prob
        self.initial_modifier_prob = initial_modifier_prob
        self.modifier_jump_prob = modifier_jump_prob
        self.delta_t = delta_t
        self.c_I = c_I
        self.c_P = c_P
        self.c0_I = c0_I
        self.c0_F = c0_F

        minor_observation_space = Discrete(2)
        minor_action_space = Discrete(2)
        major_observation_space = Discrete(2)
        major_action_space = Discrete(2)

        mu_0 = np.array([1 - initial_infection_prob, initial_infection_prob])
        mu_0_major = np.array([1 - initial_modifier_prob, initial_modifier_prob])

        super().__init__(minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents, mu_0, mu_0_major, **kwargs)

    def next_states(self, t, xs, y, us, u_major):
        mu_infected = np.mean(xs)
        recoveries = np.random.rand(self.num_agents) < self.recovery_rate * self.delta_t
        infections = np.random.rand(self.num_agents) < self.infection_rate * mu_infected * (0.5 + y + (u_major == 0)) * self.delta_t
        new_xs = xs * (1-recoveries) \
                 + (1-xs) * (1-us) * infections \
                 + (1-xs) * us * 0

        new_y = (1-y) if np.random.rand() < self.modifier_jump_prob * self.delta_t else y

        return new_xs, new_y

    def get_P(self, mu_disc):
        P = np.zeros((self.minor_action_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_observation_space.n, self.minor_observation_space.n))

        P[..., 1, 0] = self.recovery_rate * self.delta_t
        P[..., 1, 1] = 1 - P[..., 1, 0]

        P[0, 0, 0, ..., 0, 1] = np.minimum(1, self.infection_rate * mu_disc[1] * 1.5 * self.delta_t)
        P[0, 1, 0, ..., 0, 1] = np.minimum(1, self.infection_rate * mu_disc[1] * 2.5 * self.delta_t)
        P[0, 0, 1, ..., 0, 1] = np.minimum(1, self.infection_rate * mu_disc[1] * 0.5 * self.delta_t)
        P[0, 1, 1, ..., 0, 1] = np.minimum(1, self.infection_rate * mu_disc[1] * 1.5 * self.delta_t)
        P[1, ..., 0, 1] = 0
        P[..., 0, 0] = 1 - P[..., 0, 1]

        return P

    def get_P_0(self, mu_disc):
        P_major = np.zeros((self.major_observation_space.n, self.major_action_space.n, mu_disc.shape[-1],
                            self.major_observation_space.n))

        P_major[0, ..., 1] = self.modifier_jump_prob * self.delta_t
        P_major[0, ..., 0] = 1 - P_major[0, ..., 1]
        P_major[1, ..., 0] = self.modifier_jump_prob * self.delta_t
        P_major[1, ..., 1] = 1 - P_major[1, ..., 0]

        return P_major

    def reward(self, t, xs, y, us, u_major):
        minor_rewards = - self.c_I * xs - self.c_P * us * (0.5 + u_major)
        major_reward = - self.c0_I * np.mean(xs) - self.c0_F * u_major * (0.5 - np.mean(xs))
        return minor_rewards, major_reward

    def get_R(self, mu_disc):
        R = np.zeros((self.minor_observation_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_action_space.n))
        R[:, :, 0, ..., 1] -= self.c_P * 0.5
        R[:, :, 1, ..., 1] -= self.c_P * 1.5
        R[1, ...] -= self.c_I
        return R

    def get_R_0(self, mu_disc):
        R_0 = np.zeros((self.major_observation_space.n, mu_disc.shape[-1], self.major_action_space.n))
        R_0[..., 1] -= self.c0_F * (0.5 - np.expand_dims(mu_disc[1], (0,)))
        R_0 -= self.c0_I * np.expand_dims(mu_disc[1], (0, 2))
        return R_0
