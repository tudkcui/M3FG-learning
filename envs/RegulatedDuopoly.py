import itertools

import numpy as np
from gym.spaces import Discrete

from envs.m3fg import MajorMinorMARLEnv


class RegulatedDuopoly(MajorMinorMARLEnv):
    """
    Models the Duopoly problem.
    """

    def __init__(self, change_rate: float = 0.7, major_change_rate: float = 0.3, time_steps: int = 100,
                 num_quality_diffs: int = 5,
                 c_quality: float = 0.75, c_move: float = 1.0,
                 c0_balance: float = 1, c0_act: float = 1,
                 num_agents: int = 300, **kwargs):
        self.change_rate = change_rate
        self.major_change_rate = major_change_rate
        self.num_quality_diffs = num_quality_diffs
        self.num_states = self.num_quality_diffs * 2 + 1
        self.c_quality = c_quality
        self.c_move = c_move
        self.c0_balance = c0_balance
        self.c0_act = c0_act

        minor_observation_space = Discrete(2)
        minor_action_space = Discrete(2)
        major_observation_space = Discrete(self.num_states)
        major_action_space = Discrete(3)

        mu_0 = np.array([1, 0,])
        mu_0_major = np.array([1 / (self.num_quality_diffs * 2 + 1), ]
                              * (self.num_quality_diffs * 2 + 1))

        super().__init__(minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents, mu_0, mu_0_major, **kwargs)

    def next_states(self, t, xs, y, us, u_major):
        mu = np.mean([xs == loc for loc in range(2)], axis=1)

        changes_plus = np.random.rand(self.num_agents) < self.change_rate * (0.2 + mu[1])
        changes_minus = np.random.rand(self.num_agents) < self.change_rate * (0.2 + mu[0])
        new_xs = (xs == us) * xs \
                 + (xs < us) * (us * changes_plus + xs * (1 - changes_plus)) \
                 + (xs > us) * (us * changes_minus + xs * (1 - changes_minus))

        change_plus = np.random.rand(self.num_agents) < (0.1 + mu[1]) * self.major_change_rate \
                      * (y + 1 + u_major * 4) / self.num_quality_diffs / 2
        change_minus = np.random.rand(self.num_agents) < (0.1 + mu[0] - u_major) * self.major_change_rate \
                       * (self.num_states - y + u_major * 4) / self.num_quality_diffs / 2
        new_y = np.clip(y + change_plus - change_minus, self.num_quality_diffs)

        return new_xs, new_y

    def get_P(self, mu_disc):
        P = np.zeros((self.minor_action_space.n, self.major_observation_space.n, self.major_action_space.n,
                      mu_disc.shape[-1], self.minor_observation_space.n, self.minor_observation_space.n))

        P[0, 0, 0] += 1
        P[1, 0, 0] += 1 - self.change_rate * (0.2 + np.expand_dims(mu_disc[1], axis=(0,1,2,4,5)))
        P[1, 0, 1] += self.change_rate * (0.2 + np.expand_dims(mu_disc[1], axis=(0,1,2,4,5)))

        P[1, 1, 1] += 1
        P[0, 1, 0] += self.change_rate * (0.2 + np.expand_dims(mu_disc[0], axis=(0,1,2,4,5)))
        P[0, 1, 1] += 1 - self.change_rate * (0.2 + np.expand_dims(mu_disc[0], axis=(0,1,2,4,5)))

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
            R[loc, ...] += self.c_food * np.expand_dims(fillings_all[:, loc], axis=(1, 2))

            R[loc, ...] -= self.c_crowd * np.expand_dims(mu_disc[loc], axis=(0, 2))

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
