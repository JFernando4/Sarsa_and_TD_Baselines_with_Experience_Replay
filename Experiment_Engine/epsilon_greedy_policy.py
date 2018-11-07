from numpy.random import uniform, randint
from numpy import array, zeros
import numpy as np

from Experiment_Engine.config import Config
from Experiment_Engine.util import check_attribute_else_default


class EpsilonGreedyPolicy:

    def __init__(self, config=None):
        """
        Parameters in config:
        Name:               Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions         int             3                   Number of actions available to the agent
        epsilon             float           0.1                 Epsilon before annealing
        """
        self.config = config or Config()
        assert isinstance(config, Config)
        self.num_actions = check_attribute_else_default(self.config, 'num_actions', 3)
        self.epsilon = check_attribute_else_default(self.config, 'epsilon', 0.1)
        self.p_random = (self.epsilon / self.num_actions)
        self.p_optimal = self.p_random + (1 - self.epsilon)

    """ Chooses an action from q according to the probability epsilon"""

    def choose_action(self, q_value):
        p = uniform()
        if True in (np.array(q_value) == np.inf):
            raise ValueError("One of the Q-Values has a value of infinity.")
        if p < self.epsilon:
            action = randint(self.num_actions, dtype=np.uint8)
        else:
            # choosing a random action from all the possible maximum action
            action = np.uint8(np.random.choice(np.argwhere(q_value == np.max(q_value)).flatten(), size=1)[0])
        return action

    """" Returns the probability of a given action or of all the actions """

    def probability_of_action(self, q_values, action=0, all_actions=False):
        assert isinstance(q_values, np.ndarray)
        max_q = np.max(q_values)
        total_max_actions = np.sum(max_q == array(q_values))
        action_probabilities = zeros(self.num_actions, dtype=np.float64) + self.p_random

        """ Sanity Check:
        Let p_random = epsilon / (#actions), p_optimal = p_random + (1 - epsilon), and (#optimal) and (#actions) 
        be the total number of optimal actions and total number of actions, respectively. Then we have:

        \sum_{a != optimal} p_random + \sum_{a == optimal} [(p_optimal + (#optimal -1) p_random] / (#optimal)   =
            p_random * (#actions - #optimal) + (#optimal) [p_optimal + (#optimal-1) p_random] / (#optimal)      =
            (#actions) p_random) + p_optimal - p_random = epsilon + (1-epsilon) + p_random - p_random = 1

        as long as epsilon \in [0,1]
        """
        action_probabilities[np.squeeze(np.argwhere(q_values == max_q))] = \
            (self.p_optimal + (total_max_actions - 1) * self.p_random) / total_max_actions

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]

    def batch_probability_of_action(self, q_values):
        """
        Returns (row-wise) the probability of selecting the actions corresponding to each q_values
        Assumption: q_values has shape [batch_size, number_of_actions]
        """
        max_qs = np.max(q_values, axis=1)
        equal_to_max_qs = np.equal(q_values, max_qs[:, None])
        total_max_actions = np.sum(equal_to_max_qs, axis=1)
        max_actions_indices = np.argwhere(equal_to_max_qs)

        action_probabilities = np.zeros(q_values.shape, dtype=np.float64) + self.p_random
        action_probabilities[max_actions_indices[:, 0], max_actions_indices[:, 1]] = \
            np.divide(self.p_optimal + (total_max_actions[max_actions_indices[:, 0]] - 1) * self.p_random,
                      total_max_actions[max_actions_indices[:, 0]])
        return action_probabilities
