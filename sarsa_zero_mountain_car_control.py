import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
import time

# Environment
from Experiment_Engine import MountainCar
# Function Approximator
from Experiment_Engine import ExperienceReplayBuffer, NeuralNetworkFunctionApproximator, FullyConnectedModel
# Renforcement Learning Agent
from Experiment_Engine import SarsaZeroAgent, SarsaZeroReturnFunction, EpsilonGreedyPolicy
# Parameters
from Experiment_Engine import Config

MAX_EPISODES = 500


class ExperimentAgent():
    def __init__(self, experiment_parameters):

        """ Experiment Parameters """
        self.tnetwork_update_freq = experiment_parameters.tnetwork_update_freq
        self.alpha = experiment_parameters.alpha
        self.hidden_units = experiment_parameters.hidden_units
        self.xavier_init = experiment_parameters.xavier_initialization
        self.max_steps = experiment_parameters.max_steps

        self.tf_sess = tf.Session()

        """ Experiment COnfiguration """
        self.config = Config()
        self.summary = {}
        self.config.save_summary = True

        """ Environment Parameters """
        self.config.max_actions = self.max_steps
        self.config.num_actions = 3             # Number of actions in Mountain Car
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
        self.config.obs_dtype = np.float32      # Data type of the raw environment's states

        """ Policy Parameters """
        self.config.epsilon = 0.1
        self.config.onpolicy = True

        """ RL Agent Parameters """
        self.config.gamma = 1
        self.config.er_start_size = 1000
        self.config.er_init_steps_count = 0
        self.config.fixed_tpolicy = False

        " Environment "
        self.env = MountainCar(config=self.config, summary=self.summary)

        " Models "
        self.tnetwork = FullyConnectedModel(config=self.config, name='target')          # Target Network
        self.unetwork = FullyConnectedModel(config=self.config, name='update')          # Update Network

        """ Policies """
        self.target_policy = EpsilonGreedyPolicy(config=self.config)

        """ Sarsa Zero Return Function """
        self.rl_return_fun = SarsaZeroReturnFunction(tpolicy=self.target_policy, config=self.config)

        """ Experience Replay Buffer"""
        self.er_buffer = ExperienceReplayBuffer(config=self.config, return_function=self.rl_return_fun)

        """ Neural Network """
        self.function_approximator = NeuralNetworkFunctionApproximator(optimizer=self.optimizer,
                                                                       target_network=self.tnetwork,
                                                                       update_network=self.unetwork,
                                                                       er_buffer=self.er_buffer,
                                                                       config=self.config,
                                                                       tf_session=self.tf_sess,
                                                                       summary=self.summary)

        """ RL Agent """
        self.agent = SarsaZeroAgent(environment=self.env, function_approximator=self.function_approximator,
                                    behaviour_policy=self.target_policy, er_buffer=self.er_buffer, config=self.config,
                                    summary=self.summary)

        # number_of_parameters = 0
        # for variable in self.tnetwork.get_variables_as_list(self.tf_sess):
        #     number_of_parameters += np.array(variable).flatten().size
        # print("The number of parameters in the network is:", number_of_parameters)  # Answer: 6003

    def train(self):
        self.agent.train(num_episodes=1)
        self.function_approximator.store_summary()
        self.env.store_summary()
        self.agent.store_summary()

    def get_number_of_steps(self):
        return np.sum(self.summary['steps_per_episode'])

    def get_episode_number(self):
        return len(self.summary['steps_per_episode'])

    def get_train_data(self):
        return_per_episode = self.summary['return_per_episode']
        nn_loss = self.summary['cumulative_loss']
        return return_per_episode, nn_loss

    def save_results(self, dir_name):
        env_info = np.cumsum(self.summary['steps_per_episode'])
        return_per_episode = self.summary['return_per_episode']
        total_loss_per_episode = self.summary['cumulative_loss']
        results = {'return_per_episode': return_per_episode, 'env_info': env_info,
                   'total_loss_per_episode': total_loss_per_episode}
        with open(os.path.join(dir_name, 'results.p'), mode="wb") as results_file:
            pickle.dump(results, results_file)

    def save_parameters(self, dir_name):
        txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        params_txt = open(txt_file_pathname, "w")
        params_txt.write("# Agent: Sarsa Zero #\n")
        params_txt.write("\tgamma = " + str(self.rl_return_fun.gamma) + "\n")
        params_txt.write("\ton policy = " + str(self.rl_return_fun.onpolicy) + "\n")
        params_txt.write("\n")

        params_txt.write("# Target Policy #\n")
        params_txt.write("\tepsilon = " + str(self.target_policy.epsilon) + "\n")
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
    def __init__(self, experiment_parameters, results_dir=None, max_number_of_episodes=500):
        self.agent = ExperimentAgent(experiment_parameters=experiment_parameters)
        self.results_dir = results_dir
        self.max_number_of_episodes = max_number_of_episodes
        self.agent.save_parameters(self.results_dir)

        if max_number_of_episodes > MAX_EPISODES:
            raise ValueError

    def run_experiment(self, verbose=True):
        while self.agent.get_episode_number() < self.max_number_of_episodes:
            if verbose:
                print("\nTraining episode", str(len(self.agent.get_train_data()[0]) + 1) + "...")
            self.agent.train()
            if verbose:
                return_per_episode, nn_loss = self.agent.get_train_data()
                if len(return_per_episode) < 50:
                    print("The average return is:", np.average(return_per_episode))
                    print("The average training loss is:", np.average(nn_loss))
                else:
                    print("The average return is:", np.average(return_per_episode[-50:]))
                    print("The average training loss is:", np.average(nn_loss[-50:]))
                print("The return in the last episode was:", return_per_episode[-1])
                print("The total number of steps is:", self.agent.get_number_of_steps())
                print("The total average return is:", np.average(return_per_episode))
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-quiet', action='store_false', default=True)
    parser.add_argument('-name', action='store', default='agent_1', type=str)
    parser.add_argument('-episodes', action='store', default=MAX_EPISODES - 1, type=np.int32)
    parser.add_argument('-tnetwork_update_freq', action='store', default=1000, type=np.int32)
    parser.add_argument('-alpha', action='store', default=0.00025, type=np.float64)
    parser.add_argument('-onpolicy', action='store_true', default=False)
    parser.add_argument('-hidden_units', action='store', default=800, type=np.int64)
    parser.add_argument('-xavier_initialization', action='store_true', default=False)
    parser.add_argument('-max_steps', action='store', default=1000, type=np.int32)
    args = parser.parse_args()

    """ Directories """
    working_directory = os.getcwd()
    results_directory = os.path.join(working_directory, "Results", "Mountain_Car_Control")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    run_results_directory = os.path.join(results_directory, args.name)
    if not os.path.exists(run_results_directory):
        os.makedirs(run_results_directory)

    exp_params = args
    experiment = Experiment(results_dir=run_results_directory, max_number_of_episodes=args.episodes,
                            experiment_parameters=exp_params)
    start_time = time.time()
    experiment.run_experiment(verbose=args.quiet)
    end_time = time.time()
    print("Total running time:", end_time - start_time)
