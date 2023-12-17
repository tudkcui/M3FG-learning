from abc import ABC, abstractmethod

import numpy as np


class MajorMinorMARLEnv(ABC):
    """
    Models a major-minor mean-field MARL problem in discrete time.
    """

    def __init__(self, minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                 time_steps, num_agents, mu_0, mu_0_major, **kwargs):
        self.minor_observation_space = minor_observation_space
        self.minor_action_space = minor_action_space
        self.major_observation_space = major_observation_space
        self.major_action_space = major_action_space
        self.time_steps = time_steps
        self.num_agents = num_agents
        self._agent_ids = list(range(self.num_agents))
        self.mu_0 = mu_0
        self.mu_0_major = mu_0_major

        super().__init__()

        self.xs = None
        self.y = None
        self.t = None

    def reset(self):
        self.t = 0
        self.xs = self.sample_initial_minor_states()
        self.y = self.sample_initial_major_state()
        return self.get_observation()

    def get_observation(self):
        return self.xs, self.y

    def action_space_sample(self, agent_ids: list = None):
        out = {i: self.minor_action_space.sample() for i in range(self.num_agents)}
        out['major'] = self.major_action_space.sample()
        return out

    def sample_initial_minor_states(self):
        return np.random.choice(range(len(self.mu_0)), size=(self.num_agents,), p=self.mu_0)

    def sample_initial_major_state(self):
        return np.random.choice(range(len(self.mu_0_major)), p=self.mu_0_major)

    def step(self, action_minor, action_major):
        next_xs, next_y = self.next_states(self.t, self.xs, self.y, action_minor, action_major)
        rewards = self.reward(self.t, self.xs, self.y, action_minor, action_major)

        self.t += 1
        self.xs = next_xs
        self.y = next_y
        self.update()

        return self.get_observation(), rewards, self.t >= self.time_steps - 1, {}

    """
    Note that for fast execution, we vectorize and use the states and actions of all agents directly. 
     The implementing class makes sure that the next states and reward function follow the MFC model assumptions. """
    @abstractmethod
    def next_states(self, t, xs, y, us, u_major):
        pass  # sample new states for all agents

    @abstractmethod
    def reward(self, t, xs, y, us, u_major):
        pass  # sample reward defined on the state-action mean-field

    def update(self):
        pass

    """ For discretization """
    def get_P(self, mu_disc):
        pass  # Return joint transition matrices over actions U on X x X^0 x |mu_disc|

    def get_P_0(self, mu_disc):
        pass  # Return joint transition matrices over actions U^0 on X^0 x |mu_disc|

    def get_R(self, mu_disc):
        pass  # Return array X x X^0 x U^0 x |mu_disc| x U of expected rewards

    def get_R_0(self, mu_disc):
        pass  # Return array X^0 x |mu_disc| x U^0 of expected rewards

    def get_reward_vec_minor(self, mu_disc, action_probs_minor, action_probs_major):
        R = self.get_R(mu_disc)

        r""" Marginalize out major actions, \sum_u^0 P(...|u^0) pi(u^0\...) """
        R_marginalized = np.sum(R * np.expand_dims(action_probs_major.transpose((0, 2, 1)), axis=(0, 4)), axis=2)

        return R_marginalized  # Return array X x X^0 x |mu_disc| x U of expected rewards

    def get_transition_probs_minor(self, mu_disc, action_probs_minor, action_probs_major):
        """ First, get the transition prob matrix of the major state and mf """
        P_major = self.get_transition_probs_major(mu_disc, action_probs_minor)

        """ Then, get the transition probs of the representative agent """
        P = self.get_P(mu_disc)

        """ Fuse transition matrices to form transition matrix on X x X^0 x |mu_disc| for given action U """
        P_total = np.expand_dims(P_major, axis=(0, 3, 4)) * np.expand_dims(P.transpose((4, 1, 3, 0, 5, 2)), axis=(-2, -1))

        """ Marginalize out major actions with given probs """
        P_final = np.sum(P_total * np.expand_dims(action_probs_major, axis=(0, 3, 4, 6, 7)), axis=5)

        return P_final  # Return joint transition matrices over actions U on X x X^0 x |mu_disc|

    def get_reward_vec_major(self, mu_disc, action_probs_minor):
        R_0 = self.get_R_0(mu_disc)
        return R_0  # Return array X^0 x |mu_disc| x U^0 of rewards

    def get_transition_probs_major(self, mu_disc, action_probs_minor):
        r""" Define the minor transition prob matrix """
        P = self.get_P(mu_disc)

        r""" Marginalize out self.minor_action_space from current action probs, \sum_u P(...|u) pi(u\...) """
        P_marginalized = np.sum(P * np.expand_dims(action_probs_minor.transpose((3, 1, 2, 0)), axis=(2, 5)), axis=0)

        r""" Compute next mu for any given current mu and configuration in P """
        mu_nexts = np.diagonal(np.einsum('ijkmn,mo->ijkno', P_marginalized, mu_disc), axis1=2, axis2=4)
        # mu_nexts = np.diagonal(np.matmul(np.swapaxes(P_marginalized, axis1=-1, axis2=-2), mu_disc), axis1=2, axis2=4)

        r""" Project back to epsilon net in L1 dist """
        dists_to_net = np.sum(np.abs(np.expand_dims(mu_nexts, axis=-1)
                                    - np.expand_dims(mu_disc, axis=(0, 1, 3))), axis=2)
        mu_closest_indices = dists_to_net.argmin(-1)

        r""" Write the singular transition prob matrix (by deterministic T) from the indices """
        P_mf = np.zeros(mu_closest_indices.shape + (mu_disc.shape[-1],))
        for ijk, x in np.ndenumerate(mu_closest_indices):
            P_mf[ijk + (x,)] = 1

        r""" Define the major transition prob matrix """
        P_major = self.get_P_0(mu_disc)

        r""" Fuse both together to obtain transition matrix on M3FG state, 
        i.e. X^0 \times P(X) \times U^0 \to P(X^0 \times P(X)) """
        P_total = np.swapaxes(np.expand_dims(P_major, axis=-1) * np.expand_dims(P_mf, axis=-2), 1, 2)

        return P_total
