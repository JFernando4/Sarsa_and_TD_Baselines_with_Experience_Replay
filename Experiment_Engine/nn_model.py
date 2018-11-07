import tensorflow as tf
import numpy as np

from Experiment_Engine.config import Config
from Experiment_Engine.util import check_attribute_else_default


def linear_transfer(x):
    return x


def fully_connected(name, label, var_in, dim_in, dim_out, initializer, transfer, reuse=False, xavier_init=True):
    """Standard fully connected layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                if xavier_init:
                    W = tf.get_variable("W", [dim_in, dim_out], initializer=initializer)
                    b = tf.get_variable("b", [dim_out], initializer=initializer)
                else:
                    W = tf.get_variable("W", [dim_in, dim_out],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
                    b = tf.get_variable("b", [dim_out],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

    z_hat = tf.matmul(var_in, W)
    z_hat = tf.nn.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


"""
Creates a model with k fully connected layers followed by one linear output layer for action values
"""
class ActionValueFullyConnectedModel:

    def __init__(self, config=None, name="default", SEED=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        obs_dims                list            [2]                 the dimensions of the observations seen by the agent
        num_actions             int             3                   the number of actions available to the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        full_layers             int             3                   number of fully connected layers
        xavier_init             bool            True                whether to use a variant of xavier initialization
                                                                    otherwise, matrices are initialized according to
                                                                    N(0, 0.5) and bias are initialized according to
                                                                    N(0, 0.1) 
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.obs_dims = check_attribute_else_default(config, 'obs_dims', [2])
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.full_layers = check_attribute_else_default(config, 'full_layers', 3)
        self.xavier_init = check_attribute_else_default(config, 'xavier_init', True)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name
        tf.get_collection(self.name)

        " Dimensions "
        dim_in = [np.prod(self.obs_dims)] + self.dim_out[:-1]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, dim_in[0]))             # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        " Variables for Training "
        self.train_vars = []

        " Fully Connected Layers "
        current_y_hat = self.x_frames
        for j in range(self.full_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = fully_connected(
                self.name, "full_" + str(j + 1), current_y_hat, dim_in[j], self.dim_out[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[j]), seed=SEED), self.gate_fun,
                xavier_init=self.xavier_init)
            current_y_hat = y_hat
            tf.add_to_collection(self.name, W)
            tf.add_to_collection(self.name, b)
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], self.num_actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer,
            xavier_init=self.xavier_init)
        tf.add_to_collection(self.name, W)
        tf.add_to_collection(self.name, b)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_tensor(self):
        return self.train_vars[0]

"""
Creates a model with k fully connected layers followed by one linear output layer for state values
"""
class StateValueFullyConnectedModel:

    def __init__(self, config=None, name="default", SEED=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        obs_dims                list            [2]                 the dimensions of the observations seen by the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        full_layers             int             3                   number of fully connected layers
        xavier_init             bool            True                whether to use a variant of xavier initialization
                                                                    otherwise, matrices are initialized according to
                                                                    N(0, 0.5) and bias are initialized according to
                                                                    N(0, 0.1) 
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.obs_dims = check_attribute_else_default(config, 'obs_dims', [2])
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.full_layers = check_attribute_else_default(config, 'full_layers', 3)
        self.xavier_init = check_attribute_else_default(config, 'xavier_init', True)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name
        tf.get_collection(self.name)

        " Dimensions "
        dim_in = [np.prod(self.obs_dims)] + self.dim_out[:-1]
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, dim_in[0]))             # input frames
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        " Variables for Training "
        self.train_vars = []

        " Fully Connected Layers "
        current_y_hat = self.x_frames
        for j in range(self.full_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = fully_connected(
                self.name, "full_" + str(j + 1), current_y_hat, dim_in[j], self.dim_out[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[j]), seed=SEED), self.gate_fun,
                xavier_init=self.xavier_init)
            current_y_hat = y_hat
            tf.add_to_collection(self.name, W)
            tf.add_to_collection(self.name, b)
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], 1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer,
            xavier_init=self.xavier_init)
        tf.add_to_collection(self.name, W)
        tf.add_to_collection(self.name, b)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = self.y_hat
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_tensor(self):
        return self.train_vars[0]