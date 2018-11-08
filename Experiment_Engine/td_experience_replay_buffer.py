import numpy as np

from Experiment_Engine.circular_buffer import CircularBuffer
from Experiment_Engine.td_zero_agent_with_fixed_policy import TDZeroReturnFunction
from Experiment_Engine.util import check_attribute_else_default
from Experiment_Engine.config import Config


class TDExperienceReplayBuffer:

    def __init__(self, config, return_function):

        """ Parameters:
        Name:               Type:           Default:            Description: (Omitted when self-explanatory)
        buff_sz             int             10                  buffer size
        batch_sz            int             1
        env_state_dims      list            [2,2]               dimensions of the observations to be stored in the buffer
        obs_dtype           np.type         np.uint8            the data type of the observations
        """
        assert isinstance(config, Config)
        self.config = config
        self.buff_sz = check_attribute_else_default(self.config, 'buff_sz', 10)
        self.batch_sz = check_attribute_else_default(self.config, 'batch_sz', 1)
        self.env_state_dims = list(check_attribute_else_default(self.config, 'env_state_dims', [2,2]))
        self.obs_dtype = check_attribute_else_default(self.config, 'obs_dtype', np.uint8)

        """ Parameters for Return Function """
        assert isinstance(return_function, TDZeroReturnFunction)
        self.return_function = return_function

        """ Parameters to keep track of the current state of the buffer """
        self.current_index = 0
        self.full_buffer = False

        """ Circular Buffers """
        self.state = CircularBuffer(self.buff_sz, shape=tuple(self.env_state_dims), dtype=self.obs_dtype)
        self.reward = CircularBuffer(self.buff_sz, shape=(), dtype=np.int32)
        self.terminate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.timeout = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.estimated_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float64)
        self.up_to_date = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)

    def store_observation(self, observation):
        """ The only two keys that are required are 'state' """
        assert isinstance(observation, dict)
        assert all(akey in observation.keys() for akey in ["reward", "state", "terminate", "bprobabilities"])

        temp_terminate = observation['terminate']
        temp_timeout = observation['timeout']
        reward = observation["reward"]

        self.state.append(observation["state"])
        self.reward.append(reward)
        self.terminate.append(temp_terminate)
        self.timeout.append(temp_timeout)
        self.estimated_return.append(0.0)
        self.up_to_date.append(False)

        self.current_index += 1
        if self.current_index >= self.buff_sz:
            self.current_index = 0
            self.full_buffer = True

    def sample_indices(self):
        bf_start = self.terminate.start
        inds_start = 0
        if not self.full_buffer:
            inds_end = self.current_index - 1
        else:
            last_buffer_index = self.buff_sz - 1
            inds_end = last_buffer_index - 1
        sample_inds = np.random.randint(inds_start, inds_end, size=self.batch_sz)
        terminations_timeout = np.logical_or(self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap'),
                                             self.timeout.data.take(bf_start + sample_inds, axis=0, mode='wrap'))
        terminations_timeout_sum = np.sum(terminations_timeout)
        while terminations_timeout_sum != 0:
            bad_inds = np.squeeze(np.argwhere(terminations_timeout))
            new_inds = np.random.randint(inds_start, inds_end, size=terminations_timeout_sum)
            sample_inds[bad_inds] = new_inds
            terminations_timeout = np.logical_or(self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap'),
                                                 self.timeout.data.take(bf_start + sample_inds, axis=0, mode='wrap'))
            terminations_timeout_sum = np.sum(terminations_timeout)
        return sample_inds

    def get_data(self, update_function):
        indices = self.sample_indices()
        bf_start = self.state.start

        sample_states = self.state.data.take(bf_start + indices, axis=0, mode='wrap')

        estimated_returns = np.zeros(self.batch_sz, dtype=np.float64)

        """ Retrieving the estimated returns that are up to date """
        utd_returns = self.up_to_date.data.take(bf_start + indices, axis=0, mode='wrap')  # utd = up to date
        utd_batch_indices = np.squeeze(np.argwhere(utd_returns), axis=1)
        utd_buffer_indices = indices[utd_batch_indices]
        estimated_returns[utd_batch_indices] = \
            self.estimated_return.data.take(bf_start + utd_buffer_indices, axis=0, mode='wrap')

        """ Computing the estimated returns that are not up to date """
        not_utd_batch_indices = np.squeeze(np.argwhere(np.logical_not(utd_returns)), axis=1)
        not_utd_buffer_indices = indices[not_utd_batch_indices]
        next_time_step_indices = bf_start + not_utd_buffer_indices + 1
        adjusted_batch_sz = self.batch_sz - np.array(utd_batch_indices).size

        if adjusted_batch_sz > 0:
            next_state = self.state.data.take(next_time_step_indices, axis=0, mode='wrap')
            next_state_values = update_function(next_state, reshape=False).reshape([adjusted_batch_sz])
            next_reward = self.reward.data.take(next_time_step_indices, axis=0, mode='wrap')
            next_timeout = self.timeout.data.take(next_time_step_indices, axis=0, mode='wrap')
            next_termination = self.terminate.data.take(next_time_step_indices, axis=0, mode='wrap')

            estimated_returns[not_utd_batch_indices] = \
                self.return_function.batch_return_function(next_reward, next_state_values, next_termination,
                                                           next_timeout, adjusted_batch_sz)

            self.estimated_return.data.put(indices=bf_start + not_utd_buffer_indices,
                                           values=estimated_returns[not_utd_batch_indices], mode='wrap')
            self.up_to_date.data.put(indices=bf_start + not_utd_buffer_indices, values=True, mode='wrap')

        return sample_states, estimated_returns

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - 1)

    def out_of_date(self):
        self.up_to_date.data[:] = False
