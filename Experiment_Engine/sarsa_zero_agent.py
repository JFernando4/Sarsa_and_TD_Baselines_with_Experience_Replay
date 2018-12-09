from numpy import inf
import numpy as np

from Experiment_Engine.util import check_attribute_else_default, check_dict_else_default
from Experiment_Engine.config import Config


class SarsaZeroAgent:

    def __init__(self, environment, function_approximator, behaviour_policy, er_buffer, config=None,
                 summary=None, reshape=True):
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
        fixed_tpolicy           bool            False               whether the policy is fixed (e.g., a function of
                                                                    the state) or changes over time 
                                                                    (e.g., epsilon-greedy or a function of the q-values)
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
        # Whether to reshape the mountain car observations
        self.reshape = reshape

    def train(self, num_episodes):
        if num_episodes == 0: return

        for episode in range(num_episodes):
            # Current State, Action, and Q_values
            S = self.env.get_current_state()
            q_values = self.fa.get_next_states_values(self.scale_state(S))

            if self.fixed_tpolicy:
                A = self.bpolicy.choose_action(S)
                if self.config.er_init_steps_count < self.er_start_size:
                    self.config.er_init_steps_count += 1
                bprob = 1
            else:
                if self.config.er_init_steps_count >= self.er_start_size:
                    A = self.bpolicy.choose_action(q_values)
                    bprob = self.bpolicy.probability_of_action(q_values, action=A, all_actions=False)
                else:
                    A = np.random.randint(len(q_values))
                    bprob = 1 / len(q_values)
                    self.config.er_init_steps_count += 1

            # Storing in the experience replay buffer
            observation = {"reward": np.float64(0), "action": A, "state": self.scale_state(S),
                           "terminate": False, "bprobabilities": np.float64(bprob), "timeout": False}
            self.er_buffer.store_observation(observation)

            T = inf
            t = 0

            while t != T:
                # Step in the environment
                S, R, terminate, timeout = self.env.update(A)
                q_values = self.fa.get_next_states_values(self.scale_state(S))
                # Record Keeping
                self.cumulative_reward += R

                if terminate:
                    T = t + 1
                    A = np.uint8(0)
                else:
                    if timeout:
                        T = t + 1

                    if self.fixed_tpolicy:
                        A = self.bpolicy.choose_action(S)
                        if self.config.er_init_steps_count < self.er_start_size:
                            self.config.er_init_steps_count += 1
                        bprob = 1
                    else:
                        if self.config.er_init_steps_count >= self.er_start_size:
                            A = self.bpolicy.choose_action(q_values)
                            bprob = self.bpolicy.probability_of_action(q_values, action=A, all_actions=False)
                        else:
                            A = np.random.randint(len(q_values))
                            bprob = 1 / len(q_values)
                            self.config.er_init_steps_count += 1

                # Storing in the experience replay buffer
                observation = {"reward": R, "action": A, "state": self.scale_state(S),
                               "terminate": terminate, "bprobabilities": np.float64(bprob),
                               "timeout": timeout}
                self.er_buffer.store_observation(observation)

                if self.config.er_init_steps_count >= self.er_start_size:
                    # Updating the function approximator
                    self.fa.update()

                t += 1
            # End of episode
            self.env.reset()

    def store_summary(self):
        if self.save_summary:
            self.summary["return_per_episode"].append(self.cumulative_reward)
            self.cumulative_reward = 0

    def scale_state(self, state):
        if self.reshape:
            temp_state = np.zeros(2, dtype=np.float64)
            temp_state[0] = 2 * ((state[0] + 1.2) / 1.7) - 1
            temp_state[1] = 2 * ((state[1] + 0.07) / 0.14) - 1
            return temp_state
        else:
            return state


class SarsaZeroReturnFunction:

    def __init__(self, tpolicy, config=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        gamma                   float           1.0                 the discount factor
        onpolicy                bool            True                whether to compute the on-policy return or the
                                                                    off-policy, i.e. compute the importance sampling
                                                                    ratio or not.
        """
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.onpolicy = check_attribute_else_default(config, 'onpolicy', True)

        """
        Other Parameters:
        tpolicy - The target policy
        """
        self.tpolicy = tpolicy

    def batch_return_function(self, next_reward, next_action, next_qvalues, next_action_bprob,
                              next_termination, next_timeout, batch_size):
        tprobabilities = self.tpolicy.batch_probability_of_action(next_qvalues)

        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size, dtype=np.uint8)
        term_ind = next_termination.astype(np.uint8)
        neg_term_ind = np.subtract(one_vector, term_ind)
        timeout_ind = next_timeout.astype(np.uint8)
        neg_timeout_ind = np.subtract(one_vector, timeout_ind)

        next_qvalues = next_qvalues[batch_idxs, next_action]
        Gt_last_term = self.gamma * (neg_term_ind * neg_timeout_ind * next_qvalues +
                                     term_ind * neg_timeout_ind * 0.0 +
                                     neg_term_ind * timeout_ind * next_qvalues)

        rho = 1.0
        if not self.onpolicy:
            next_action_tprob = tprobabilities[batch_idxs, next_action]
            rho = np.divide(next_action_tprob, next_action_bprob)

        Gt = next_reward + rho * Gt_last_term
        return Gt
