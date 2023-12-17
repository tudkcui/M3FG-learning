import numpy as np
from gym.spaces import Discrete

from envs.m3fg import MajorMinorMARLEnv


class Cyber(MajorMinorMARLEnv):
    """
    Models the Cyber game.
    """

    def __init__(self, infection_rate: float = 0.8, recovery_rate: float = 0.2, time_steps: int = 100,
                 initial_infection_prob = 0.2, initial_modifier_prob = 0.5, modifier_jump_prob: float = 0.1,
                 cost_infection_minor: float = 1.0, cost_protection: float = 0.0,
                 cost_infection_major: float = 1.0, cost_hacking_major: float = 0.5,
                 monotonic_regularization: float = 0.0, num_agents: int = 300, **kwargs):
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infection_prob = initial_infection_prob
        self.initial_modifier_prob = initial_modifier_prob
        self.modifier_jump_prob = modifier_jump_prob
        self.cost_infection_minor = cost_infection_minor
        self.cost_protection = cost_protection
        self.cost_infection_major = cost_infection_major
        self.cost_hacking_major = cost_hacking_major
        self.monotonic_regularization = monotonic_regularization

        minor_observation_space = Discrete(4)
        minor_action_space = Discrete(2)
        major_observation_space = Discrete(1)
        major_action_space = Discrete(2)

        # PS, PI, US, UI
        mu_0 = np.array([(1 - initial_infection_prob) * (1-self.initial_modifier_prob),
                         initial_infection_prob * (1-self.initial_modifier_prob),
                         (1 - initial_infection_prob) * self.initial_modifier_prob,
                         initial_infection_prob * self.initial_modifier_prob])
        mu_0_major = np.array([1])

        super().__init__(minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents, mu_0, mu_0_major, **kwargs)

    def next_states(self, t, xs, y, us, u_major):
        mu_infected = np.mean((xs == 1) + (xs == 3))
        jumps = np.random.rand(self.num_agents) < self.modifier_jump_prob
        recoveries = np.random.rand(self.num_agents) < self.recovery_rate
        infections = np.random.rand(self.num_agents) \
                     < self.infection_rate * mu_infected * (0.5 + u_major) * (0.5 + xs // 2)
        xs_infection_state = xs % 2
        new_xs_infection_state = xs_infection_state * (1-recoveries) \
                                 + (1-xs_infection_state) * infections
        new_xs = ((xs // 2 + jumps * us) % 2 * 2 + new_xs_infection_state) % 4

        new_y = 0

        return new_xs, new_y

    def reward(self, t, xs, y, us, u_major):
        minor_rewards = - self.cost_infection_minor * ((xs == 1) + (xs == 3)) - self.cost_protection * ((xs == 2) + (xs == 3))
        major_reward = self.cost_infection_major * np.mean((xs == 1) + (xs == 3)) - self.cost_hacking_major * u_major
        return minor_rewards, major_reward

    def get_P(self, mu_disc):
        P = np.zeros((self.minor_action_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_observation_space.n, self.minor_observation_space.n))

        P[0, ..., 1, 0] = self.recovery_rate
        P[0, ..., 1, 1] = 1 - self.recovery_rate
        P[0, ..., 1+2, 0+2] = self.recovery_rate
        P[0, ..., 1+2, 1+2] = 1 - self.recovery_rate

        P[1, ..., 1, 0] = self.recovery_rate * (1-self.modifier_jump_prob)
        P[1, ..., 1, 1] = (1 - self.recovery_rate) * (1-self.modifier_jump_prob)
        P[1, ..., 1, 0+2] = self.recovery_rate * self.modifier_jump_prob
        P[1, ..., 1, 1+2] = (1 - self.recovery_rate) * self.modifier_jump_prob
        P[1, ..., 1+2, 0+2] = self.recovery_rate * (1-self.modifier_jump_prob)
        P[1, ..., 1+2, 1+2] = (1 - self.recovery_rate) * (1-self.modifier_jump_prob)
        P[1, ..., 1+2, 0] = self.recovery_rate * self.modifier_jump_prob
        P[1, ..., 1+2, 1] = (1 - self.recovery_rate) * self.modifier_jump_prob

        P[0, :, 0, ..., 0, 1] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5
        P[0, :, 1, ..., 0, 1] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)
        P[0, :, 0, ..., 0, 0] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5)
        P[0, :, 1, ..., 0, 0] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5))
        P[0, :, 0, ..., 0+2, 1+2] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5
        P[0, :, 1, ..., 0+2, 1+2] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)
        P[0, :, 0, ..., 0+2, 0+2] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5)
        P[0, :, 1, ..., 0+2, 0+2] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5))

        P[1, :, 0, ..., 0, 1] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5 * (1-self.modifier_jump_prob)
        P[1, :, 1, ..., 0, 1] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5) * (1-self.modifier_jump_prob)
        P[1, :, 0, ..., 0, 0] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5) * (1-self.modifier_jump_prob)
        P[1, :, 1, ..., 0, 0] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)) * (1-self.modifier_jump_prob)
        P[1, :, 0, ..., 0, 1+2] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5 * self.modifier_jump_prob
        P[1, :, 1, ..., 0, 1+2] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5) * self.modifier_jump_prob
        P[1, :, 0, ..., 0, 0+2] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5) * self.modifier_jump_prob
        P[1, :, 1, ..., 0, 0+2] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)) * self.modifier_jump_prob
        P[1, :, 0, ..., 0+2, 1+2] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5 * (1-self.modifier_jump_prob)
        P[1, :, 1, ..., 0+2, 1+2] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5) * (1-self.modifier_jump_prob)
        P[1, :, 0, ..., 0+2, 0+2] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5) * (1-self.modifier_jump_prob)
        P[1, :, 1, ..., 0+2, 0+2] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)) * (1-self.modifier_jump_prob)
        P[1, :, 0, ..., 0+2, 1] = self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5 * self.modifier_jump_prob
        P[1, :, 1, ..., 0+2, 1] = np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5) * self.modifier_jump_prob
        P[1, :, 0, ..., 0+2, 0] = (1 - self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 0.5) * self.modifier_jump_prob
        P[1, :, 1, ..., 0+2, 0] = (1 - np.minimum(1, self.infection_rate * np.expand_dims(mu_disc[1] + mu_disc[3], axis=0) * 1.5)) * self.modifier_jump_prob

        return P

    def get_P_0(self, mu_disc):
        P_major = np.zeros((self.major_observation_space.n, self.major_action_space.n, mu_disc.shape[-1],
                            self.major_observation_space.n))

        P_major[0, ..., 0] = 1

        return P_major

    def get_R(self, mu_disc):
        R = np.zeros((self.minor_observation_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_action_space.n))
        R[1, ...] -= self.cost_infection_minor
        R[3, ...] -= self.cost_infection_minor
        R[2, ...] -= self.cost_protection
        R[3, ...] -= self.cost_protection
        return R

    def get_R_0(self, mu_disc):
        R_0 = np.zeros((self.major_observation_space.n, mu_disc.shape[-1], self.major_action_space.n))
        R_0[..., 1] -= self.cost_hacking_major
        R_0 += self.cost_infection_major * np.expand_dims(mu_disc[1] + mu_disc[3], (0, 2))
        return R_0  # Return array X^0 x |mu_disc| x U^0 of rewards
