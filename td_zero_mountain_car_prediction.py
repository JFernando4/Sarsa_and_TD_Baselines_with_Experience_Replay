import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
import parser
import time

# Environment
from Experiment_Engine import MountainCar
# Function Approximator
from Experiment_Engine import TDExperienceReplayBuffer, TDNeuralNetworkFunctionApproximator, \
    StateValueFullyConnectedModel
# Renforcement Learning Agent
from Experiment_Engine import TDZeroAgent, TDZeroReturnFunction
# Parameters
from Experiment_Engine import Config

MAX_EPISODES = 2000


class ExperimentAgent():
    def __init__(self, experiment_parameters):

        """ Experiment Parameters """
        self.tnetwork_update_freq = experiment_parameters.tnetwork_update_freq
        self.alpha = experiment_parameters.alpha
        self.hidden_units = experiment_parameters.hidden_units
        self.xavier_init = experiment_parameters.xavier_initialization
        self.max_steps = experiment_parameters.max_steps
        self.replay_start = experiment_parameters.replay_start

        self.tf_sess = tf.Session()

        """ Experiment Configuration """
        self.config = Config()
        self.summary = {}
        self.summary['root_mean_squared_value_error'] = []
        self.config.save_summary = True

        """ Environment Parameters """
        self.config.max_actions = self.max_steps
        self.config.obs_dims = [2]              # Dimensions of the observations as experienced by the agent
                                                # (This could be a transformation of the raw state)

        """ Function Approximator Parameters """
        """     Neural Network and Model Parameters    """
        self.config.dim_out = [self.hidden_units]
        self.config.xavier_init = self.xavier_init
        self.config.gate_fun = tf.nn.relu
        self.config.full_layers = 1
        self.config.alpha = self.alpha
        self.config.tnetwork_update_freq = self.tnetwork_update_freq
        self.config.update_count = 0
        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01, momentum=0.95,
                                                              centered=True)

        """     Experience Replay Buffer Parameters     """
        self.config.batch_sz = 32
        self.config.buff_sz = 20000
        self.config.env_state_dims = [2]        # Dimensions of the raw environment's states
        self.config.obs_dtype = np.float64      # Data type of the raw environment's states

        """ RL Agent Parameters """
        self.config.gamma = 1.0
        self.config.er_start_size = self.replay_start
        self.config.er_init_steps_count = 0

        " Environment "
        self.env = MountainCar(config=self.config, summary=self.summary)

        " Models "
        self.tnetwork = StateValueFullyConnectedModel(config=self.config, name='target')          # Target Network
        self.unetwork = StateValueFullyConnectedModel(config=self.config, name='update')          # Update Network

        """ Sarsa Zero Return Function """
        self.rl_return_fun = TDZeroReturnFunction(config=self.config)

        """ Experience Replay Buffer"""
        self.er_buffer = TDExperienceReplayBuffer(config=self.config, return_function=self.rl_return_fun)

        """ Neural Network """
        self.function_approximator = TDNeuralNetworkFunctionApproximator(optimizer=self.optimizer,
                                                                         target_network=self.tnetwork,
                                                                         update_network=self.unetwork,
                                                                         er_buffer=self.er_buffer,
                                                                         config=self.config,
                                                                         tf_session=self.tf_sess,
                                                                         summary=self.summary)

        """ RL Agent """
        self.agent = TDZeroAgent(environment=self.env, function_approximator=self.function_approximator,
                                 er_buffer=self.er_buffer, config=self.config,
                                 summary=self.summary)

    def train(self, database):
        self.agent.train(num_episodes=1)
        self.function_approximator.store_summary()
        self.env.store_summary()
        self.agent.store_summary()
        rmsve = self.evaluate_model(database)
        assert 'root_mean_squared_value_error' in self.summary.keys()
        self.summary['root_mean_squared_value_error'].append(rmsve)

    def evaluate_model(self, database):
        states = database[:, np.arange(2)]      # The first two columns correspond to the position and velocity
        states = 2 * np.divide(np.add(states, np.array([1.2, 0.07])), np.array([1.7, 0.14])) - 1
        estimated_value_functions = np.squeeze(self.function_approximator.get_state_values_for_evaluation(states))
        true_value_functions = database[:, 2]
        estimation_error = estimated_value_functions - true_value_functions
        squared_error = estimation_error * estimation_error
        mean_squared_error = np.sum(squared_error) / squared_error.size
        rmsve = np.sqrt(mean_squared_error)
        return rmsve

    def get_number_of_steps(self):
        return np.sum(self.summary['steps_per_episode'])

    def get_episode_number(self):
        return len(self.summary['steps_per_episode'])

    def get_train_data(self):
        return_per_episode = self.summary['return_per_episode']
        nn_loss = self.summary['cumulative_loss']
        rmsve = self.summary['root_mean_squared_value_error']
        return return_per_episode, nn_loss, rmsve

    def save_results(self, dir_name):
        env_info = np.cumsum(self.summary['steps_per_episode'])
        return_per_episode = np.array(self.summary['return_per_episode'], dtype=np.float64)
        total_loss_per_episode = np.array(self.summary['cumulative_loss'], dtype=np.float64)
        root_mean_squared_value_error = np.array(self.summary['root_mean_squared_value_error'], dtype=np.float64)
        results = {'return_per_episode': return_per_episode, 'env_info': env_info,
                   'total_loss_per_episode': total_loss_per_episode,
                   'root_mean_squared_value_error': root_mean_squared_value_error}
        with open(os.path.join(dir_name, 'results.p'), mode="wb") as results_file:
            pickle.dump(results, results_file)

    def save_parameters(self, dir_name):
        txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        params_txt = open(txt_file_pathname, "w")
        params_txt.write("# Agent: Sarsa Zero #\n")
        params_txt.write("\tgamma = " + str(self.rl_return_fun.gamma) + "\n")
        params_txt.write("\n")

        params_txt.write("# Function Approximator: Neural Network with Experience Replay #\n")
        params_txt.write("\talpha = " + str(self.function_approximator.alpha) + "\n")
        params_txt.write("\ttarget network update frequency = "
                         + str(self.function_approximator.tnetwork_update_freq) + "\n")
        params_txt.write("\tbatch size = " + str(self.er_buffer.batch_sz) + "\n")
        params_txt.write("\tbuffer size = " + str(self.er_buffer.buff_sz) + "\n")
        params_txt.write("\tfully connected layers = " + str(self.tnetwork.full_layers) + "\n")
        params_txt.write("\toutput dimensions per layer = " + str(self.tnetwork.dim_out) + "\n")
        params_txt.write("\txavier initialization = " + str(self.xavier_init) + "\n")
        params_txt.write("\tgate function = " + str(self.tnetwork.gate_fun) + "\n")
        params_txt.write("\treplay start size = " + str(self.agent.er_start_size) + "\n")
        params_txt.write("\n")
        params_txt.close()


class Experiment:
    def __init__(self, experiment_parameters, database_path, results_dir=None, max_number_of_episodes=500):
        self.agent = ExperimentAgent(experiment_parameters=experiment_parameters)
        self.results_dir = results_dir
        self.max_number_of_episodes = max_number_of_episodes
        self.agent.save_parameters(self.results_dir)
        self.database = np.load(database_path)

        if max_number_of_episodes > MAX_EPISODES:
            raise ValueError

    def run_experiment(self, verbose=True):
        while self.agent.get_episode_number() < self.max_number_of_episodes:
            if verbose:
                print("\nTraining episode", str(len(self.agent.get_train_data()[0]) + 1) + "...")
            self.agent.train(self.database)
            if verbose:
                return_per_episode, nn_loss, rmsve = self.agent.get_train_data()
                if len(return_per_episode) < 50:
                    print("The average return is:", np.average(return_per_episode))
                    print("The average training loss is:", np.average(nn_loss))
                else:
                    print("The average return is:", np.average(return_per_episode[-50:]))
                    print("The average training loss is:", np.average(nn_loss[-50:]))
                print("The return in the last episode was:", return_per_episode[-1])
                print("The root mean squared value error last episode was:", rmsve[-1])
                print("The total number of steps is:", self.agent.get_number_of_steps())
                print("The total average return is:", np.average(return_per_episode))
                print("The total average root mean squared value error is:", np.average(rmsve))
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Experiment Parameters """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-quiet', action='store_false', default=True)
    argument_parser.add_argument('-name', action='store', default='agent_1', type=str)
    argument_parser.add_argument('-episodes', action='store', default=MAX_EPISODES, type=np.int32)
    argument_parser.add_argument('-tnetwork_update_freq', action='store', default=1000, type=np.int32)
    argument_parser.add_argument('-alpha', action='store', default="0.00025", type=str)
    argument_parser.add_argument('-hidden_units', action='store', default=135, type=np.int64)
    argument_parser.add_argument('-xavier_initialization', action='store_true', default=False)
    argument_parser.add_argument('-max_steps', action='store', default=1000, type=np.int32)
    argument_parser.add_argument('-replay_start', action='store', default=1000, type=np.int32)
    args = argument_parser.parse_args()

    alpha_code = parser.expr(args.alpha).compile()
    args.alpha = eval(args.alpha)

    """ Directories """
    working_directory = os.getcwd()
    database_path = os.path.join(working_directory, 'Experiment_Engine', 'sampleOnPolicy.npy')
    assert os.path.isfile(database_path)
    results_directory = os.path.join(working_directory, "Results", "Mountain_Car_Prediction")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    run_results_directory = os.path.join(results_directory, args.name)
    if not os.path.exists(run_results_directory):
        os.makedirs(run_results_directory)

    exp_params = args
    experiment = Experiment(results_dir=run_results_directory, database_path=database_path,
                            max_number_of_episodes=args.episodes, experiment_parameters=exp_params)
    start_time = time.time()
    experiment.run_experiment(verbose=args.quiet)
    end_time = time.time()
    print("Total running time:", end_time - start_time)

    # for mountain car prediction: alpha_list_nonlinear = np.array([0.0000005, 0.000001, 0.000002, 0.000004, 0.000008,
    # 0.000016, 0.000032, 0.000064, 0.000128, 0.000256, 0.000512, 0.001024, 0.002048, 0.004096, 0.008192, 0.016384, 0.032768])
