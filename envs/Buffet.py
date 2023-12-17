import itertools

import numpy as np
from gym.spaces import Discrete

from envs.m3fg import MajorMinorMARLEnv


class Buffet(MajorMinorMARLEnv):
    """
    Models the Buffet problem.
    """

    def __init__(self, change_rate: float = 0.7, time_steps: int = 100, delta_t: float = 0.2,
                 num_locations: int = 2, max_fill_state: int = 5, gain_rate: float = 0.9, loss_rate: float = 1.0,
                 c_food: float = 0.75, c_crowd: float = 0.5, c_move: float = 1.0,
                 c0_fill: float = 2, c0_balance: float = 1,
                 num_agents: int = 300, **kwargs):
        self.change_rate = change_rate
        self.gain_rate = gain_rate
        self.loss_rate = loss_rate
        self.num_locations = num_locations
        self.fill_states = max_fill_state
        self.delta_t = delta_t
        self.c_food = c_food
        self.c_crowd = c_crowd
        self.c_move = c_move
        self.c0_fill = c0_fill
        self.c0_balance = c0_balance

        minor_observation_space = Discrete(self.num_locations)
        minor_action_space = Discrete(self.num_locations)
        major_observation_space = Discrete(self.fill_states ** self.num_locations)
        major_action_space = Discrete(self.num_locations)

        mu_0 = np.array([1,] + [0,] * (self.num_locations - 1))
        mu_0_major = np.array([1 / (self.fill_states ** self.num_locations), ]
                              * (self.fill_states ** self.num_locations))

        super().__init__(minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents, mu_0, mu_0_major, **kwargs)

    def from_fillings(self, fillings):
        return np.sum(np.expand_dims([self.fill_states ** d for d in range(self.num_locations)], axis=0) * fillings, axis=1)

    def to_fillings(self, y):
        return np.stack([(y // (self.fill_states ** d)) % self.fill_states for d in range(self.num_locations)]).transpose()

    def next_states(self, t, xs, y, us, u_major):
        changes = np.random.rand(self.num_agents) < self.change_rate * self.delta_t
        new_xs = xs * (1-changes) + us * changes

        fillings = self.to_fillings(y)

        for loc in range(self.num_locations):
            mu_loc = np.mean(xs == loc)
            loss_food = np.random.rand() < mu_loc * self.loss_rate * self.delta_t
            gain_food = np.random.rand() < self.gain_rate * self.delta_t

            fillings[loc] -= loss_food
            fillings[loc] += gain_food * (u_major == loc)

        fillings = np.clip(fillings, 0, self.fill_states - 1)
        new_y = self.from_fillings(fillings).item()

        return new_xs, new_y

    def get_P(self, mu_disc):
        P = np.zeros((self.minor_action_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_observation_space.n, self.minor_observation_space.n))

        for loc in range(self.num_locations):
            P[..., loc, loc] += 1 - self.change_rate * self.delta_t
            P[loc, ..., loc] += self.change_rate * self.delta_t

        return P

    def get_P_0(self, mu_disc):
        P_major = np.zeros((self.major_observation_space.n, self.major_action_space.n, mu_disc.shape[-1],
                            self.major_observation_space.n))

        y_all = np.array(range(self.fill_states ** self.num_locations))
        fillings_all = self.to_fillings(y_all)

        for u_major in range(self.num_locations):
            """ Case where major agent fills up loc u_major """
            fillings_gain_unclip = np.copy(fillings_all)
            fillings_gain_unclip[:, u_major] = fillings_gain_unclip[:, u_major] + 1
            prob_gain = self.gain_rate * self.delta_t

            for locs_loss in itertools.product(*[[False, True]] * self.num_locations):
                """ Case where we lose as in locs_loss, and their probs for each mu """
                probs_locs_loss = np.prod([np.expand_dims(mu_disc[loc], (0,)) * self.loss_rate * self.delta_t if loss else
                                           (1 - np.expand_dims(mu_disc[loc], (0,)) * self.loss_rate * self.delta_t)
                                           for loc, loss in enumerate(locs_loss)], axis=0)

                fillings_loss = np.copy(fillings_all)
                fillings_loss[:, locs_loss] = fillings_loss[:, locs_loss] - 1
                fillings_loss = np.clip(fillings_loss, 0, self.fill_states - 1)

                fillings_both = np.copy(fillings_gain_unclip)
                fillings_both[:, locs_loss] = fillings_both[:, locs_loss] - 1
                fillings_both = np.clip(fillings_both, 0, self.fill_states - 1)

                y_loss = self.from_fillings(fillings_loss)
                y_both = self.from_fillings(fillings_both)

                """ Each possible loss adds mass to a possible transition to loss, or both loss and gain """
                P_major[y_all, u_major, :, y_loss] += (1 - prob_gain) * probs_locs_loss
                P_major[y_all, u_major, :, y_both] += prob_gain * probs_locs_loss

        return P_major

    def reward(self, t, xs, y, us, u_major):
        fillings = self.to_fillings(y)
        mu = np.mean([xs == loc for loc in range(self.num_locations)], axis=1)
        minor_rewards = self.c_food * fillings[xs] - self.c_crowd * mu[xs] - self.c_move * (us != xs)
        major_reward = self.c0_fill * np.mean(fillings) - self.c0_balance * np.mean(np.abs([filling - np.mean(fillings) for filling in fillings]))
        return minor_rewards, major_reward

    def get_R(self, mu_disc):
        R = np.zeros((self.minor_observation_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_action_space.n))

        y_all = np.array(range(self.fill_states ** self.num_locations))
        fillings_all = self.to_fillings(y_all)

        for loc in range(self.num_locations):
            R[loc, ...] += self.c_food * np.expand_dims(fillings_all[:, loc], axis=(1, 2, 3))

            R[loc, ...] -= self.c_crowd * np.expand_dims(mu_disc[loc], axis=(0, 1, 3))

            notloc = np.delete(np.array(range(self.num_locations)), loc)
            R[loc][..., notloc] -= self.c_move

        return R

    def get_R_0(self, mu_disc):
        R_0 = np.zeros((self.major_observation_space.n, mu_disc.shape[-1], self.major_action_space.n))

        y_all = np.array(range(self.fill_states ** self.num_locations))
        fillings_all = self.to_fillings(y_all)
        mean_fillings = np.mean(fillings_all, axis=1)
        balance_fillings = np.mean(np.abs(fillings_all - np.expand_dims(mean_fillings, axis=1)), axis=1)

        R_0 += self.c0_fill * np.expand_dims(mean_fillings, axis=(1, 2))
        R_0 -= self.c0_balance * np.expand_dims(balance_fillings, axis=(1, 2))

        return R_0
