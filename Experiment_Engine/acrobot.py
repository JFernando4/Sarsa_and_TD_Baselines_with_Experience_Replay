import numpy as np
import gym

from Experiment_Engine.config import Config
from Experiment_Engine.util import check_dict_else_default, check_attribute_else_default


class Acrobot:
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 4 (theta_1, theta_2, normalized_vel_1, normalized_vel_2)
    Observation Dtype = np.float64
    Reward = -1 at every step

    Summary Name: steps_per_episode
    """

    def __init__(self, config=None, summary=None):
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             1000            The max number of actions executed before forcing
                                                                    a time out
        save_summary                bool            False           Whether to save a summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', default_value=500)
        self.save_summary = check_attribute_else_default(config, 'save_summary', default_value=False)
        self.summary = summary
        if self.save_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        " Inner state of the environment "
        self.step_count = 0
        self.openai_env = gym.make('Acrobot-v1')
        self.actions = np.array([0, 1, 2], dtype=np.int8)
        self.high = np.array([np.pi * 2, np.pi * 2, 12.56637096, 28.27433395], np.float64)
        self.low = np.array([0.0, 0.0, -12.56637096, -28.27433395], dtype=np.float64)
        self.current_state = self.reset()

    def reset(self):
        openai_state = self.openai_env.reset()
        theta_1, theta_2, normalized_vel_1, normalized_vel_2 = self.convert_openai_state(openai_state)
        self.current_state = np.array((theta_1, theta_2, normalized_vel_1, normalized_vel_2), dtype=np.float64)
        return self.current_state

    @staticmethod
    def convert_sin_cos_to_theta(cos_, sin_):
        """ Converts from cosine and sine to  an angle """
        theta1 = np.arcsin(sin_)
        theta2 = np.arccos(cos_)

        if ((np.pi / 2) >= theta1 >= 0.0) and ((np.pi / 2) >= theta2 >= 0.0):
            return theta1
        elif ((np.pi / 2) >= theta1 >= 0.0) and (np.pi >= theta2 >= (np.pi / 2)):
            return theta2
        elif (0.0 >= theta1 >= (-np.pi / 2)) and (np.pi >= theta2 >= (np.pi / 2)):
            return np.pi - theta1
        elif (0.0 >= theta1 >= (-np.pi / 2)) and ((np.pi / 2) >= theta2 >= 0.0):
            return -theta1

    def convert_openai_state(self, openai_state):
        """
        Convert the state from the Open AI format ([sin_1, cos_1, sin_2, cos_2, vel_1, vel_2]) to the format:
                            [theta_1, theta_2, normalized_vel_1, normalized_vel_2]
        The velocities are normalized between -1 and 1.
        """
        theta_1 = self.convert_sin_cos_to_theta(openai_state[0], openai_state[1])
        theta_2 = self.convert_sin_cos_to_theta(openai_state[2], openai_state[3])
        vel_1 = openai_state[4]
        normalized_vel_1 = 2 * (vel_1 - self.low[2]) / (self.high[2] - self.low[2]) - 1
        vel_2 = openai_state[5]
        normalized_vel_2 = 2 * (vel_2 - self.low[3]) / (self.high[3] - self.low[3]) - 1
        return theta_1, theta_2, normalized_vel_1, normalized_vel_2

    " Update environment "
    def update(self, A):
        self.step_count += 1

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        openai_state, reward, terminate, _ = self.openai_env.step(A)
        theta_1, theta_2, normalized_vel_1, normalized_vel_2 = self.convert_openai_state(openai_state)
        timeout = False
        if self.step_count >= self.max_actions:
            timeout = True

        self.current_state = np.array((theta_1, theta_2, normalized_vel_1, normalized_vel_2), dtype=np.float64)
        return self.current_state, reward, terminate, timeout

    def get_current_state(self):
        return self.current_state

    def store_summary(self):
        if self.save_summary:
            self.summary["steps_per_episode"].append(self.step_count)
            self.step_count = 0
