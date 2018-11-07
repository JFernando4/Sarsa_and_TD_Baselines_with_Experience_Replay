from numpy import inf
import numpy as np

from Experiment_Engine.util import check_attribute_else_default, check_dict_else_default
from Experiment_Engine.config import Config


class TDZeroAgent:

    def __init__(self, environment, function_approximator, behaviour_policy, er_buffer, config=None,
                 summary=None):
        """
        Summary Name: return_per_episode
        """
        self.config = config or Config()
        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        save_summary            bool            False               save the summary of the agent (return per episode)
        er_start_size           int             0                   number of steps sampled before training starts
        er_init_steps_count     int             0                   number of initial steps taken so far
        """
        self.save_summary = check_attribute_else_default(self.config, 'save_summary', False)
        self.er_start_size = check_attribute_else_default(self.config, 'er_start_size', 0)
        check_attribute_else_default(self.config, 'er_init_steps_count', 0)
        self.fixed_tpolicy = check_attribute_else_default(self.config, 'fixed_tpolicy', False)

        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'return_per_episode', [])

        " Other Parameters "
        # Behaviour
        self.bpolicy = behaviour_policy
        # Experience Replay Buffer
        self.er_buffer = er_buffer
        # Function Approximator: used to approximate the Q-Values
        self.fa = function_approximator
        # Environment that the agent is interacting with
        self.env = environment
        # Summaries
        self.cumulative_reward = 0

    def train(self, num_episodes):
        if num_episodes == 0: return

        for episode in range(num_episodes):
            # Current State, Action, and Q_values
            S = self.env.get_current_state()
            A = self.choose_action(S)
            # Storing in the experience replay buffer
            observation = {"reward": np.float64(0), "state": S, "terminate": False, "timeout": False}
            self.er_buffer.store_observation(observation)

            T = inf
            t = 0
            while t != T:
                # Step in the environment
                S, R, terminate, timeout = self.env.update(A)
                # Record Keeping
                self.cumulative_reward += R

                if terminate:
                    T = t + 1
                    A = np.uint8(0)
                else:
                    if timeout:
                        T = t + 1
                    A = self.choose_action(S)
                # Storing in the experience replay buffer
                observation = {"reward": R, "state": S, "terminate": terminate, "timeout": timeout}
                self.er_buffer.store_observation(observation)
                if self.config.er_init_steps_count < self.er_start_size:
                    # Populating the experience replay buffer
                    self.config.er_init_steps_count += 1
                else:
                    # Updating the function approximator
                    self.fa.update()

                t += 1
            # End of episode
            self.env.reset()

    def store_summary(self):
        if self.save_summary:
            self.summary["return_per_episode"].append(self.cumulative_reward)
            self.cumulative_reward = 0

    @staticmethod
    def choose_action(S):
        velocity = S[1]
        if velocity >= 0:
            action = 1
        else:
            action = -1
        return action


class TDZeroReturnFunction:

    def __init__(self, config=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        gamma                   float           1.0                 the discount factor
        """
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)

    def batch_return_function(self, next_reward, next_state_values, next_termination, next_timeout, batch_size):

        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size, dtype=np.uint8)
        term_ind = next_termination.astype(np.uint8)
        neg_term_ind = np.subtract(one_vector, term_ind)
        timeout_ind = next_timeout.astype(np.uint8)
        neg_timeout_ind = np.subtract(one_vector, timeout_ind)

        Gt_last_term = self.gamma * (neg_term_ind * neg_timeout_ind * next_state_values +
                                     term_ind * neg_timeout_ind * 0.0 +
                                     neg_term_ind * timeout_ind * next_state_values)

        Gt = next_reward + Gt_last_term
        return Gt
