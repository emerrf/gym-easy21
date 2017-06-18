import gym
import gym_easy21
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

np.random.seed(1)


def run_random_actions():

    easy21 = gym.make('Easy21-v0')

    for i_episode in range(100):
        log.info("Episode {}".format(i_episode))
        trace = list()
        observation, _, done, info = easy21.reset()
        while not done:
            action = easy21.action_space.sample()
            observation, reward, done, info = easy21.step(action)
            trace.append((observation, action, reward))

        log.info(info)
        log.info(trace)


class RLAlgorithm(object):
    def __init__(self, env_dim, n_0):
        self.Q = np.zeros(env_dim)
        self.N = np.zeros(env_dim)
        self.n_0 = n_0
        self.g = 1.0
        self.act_available = np.arange(env_dim[2])
        self.env_dim = env_dim

        self.wins = 0.0

    def _to_index(self, obs, action=-1):
            if action in self.act_available:
                return obs[0] - 1, obs[1] - 1, action
            else:
                return obs[0] - 1, obs[1] - 1

    def take_e_greedy_action(self, obs):
        s_n = np.sum(self.N[self._to_index(obs)])
        eps = self.n_0 / (self.n_0 + s_n)
        act_prob = np.full(self.act_available.shape,
                           float(eps) / self.act_available.size)
        act_prob[np.argmax(self.Q[self._to_index(obs)])] += (1 - eps)
        action = np.random.choice(self.act_available, p=act_prob)

        self.N[self._to_index(obs, action)] += 1.0

        return action

    def export_data(self, filename):
        with open(filename, "wb") as plk_file:
            pickle.dump(self.__dict__, plk_file)

    def import_data(self, filename):
        with open(filename, "rb") as plk_file:
            plk_content = pickle.load(plk_file)
            self.__dict__.update(plk_content)

    def plot_heatmap(self):
        plt.imshow(self.Q.max(axis=2))
        plt.colorbar()
        plt.show()
        # plt.imshow(self.N.sum(axis=2))
        # plt.colorbar()
        # plt.show()

    def plot_surface(self):

        X, Y = np.meshgrid(np.arange(self.env_dim[0]),
                           np.arange(self.env_dim[1]))
        Z = self.Q.max(axis=2).transpose()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show()


class MonteCarloAlgorithm(RLAlgorithm):
    def update_policy(self, trace):
        r = trace[-1][2]  # Use reward at the end of episode
        if r == 1:
            self.wins += 1

        for obs, a, __ in trace:
            alpha = 1.0 / self.N[self._to_index(obs, a)]
            self.Q[self._to_index(obs, a)] += (
                alpha * (self.g * r - self.Q[self._to_index(obs, a)]))


class TDLambdaAlgorithm(RLAlgorithm):
    def __init__(self, env_dim, n_0, coef_lambda):
        super(TDLambdaAlgorithm, self).__init__(env_dim, n_0)
        self.E = np.zeros(self.env_dim)
        self.reset_eligibility()
        self.coef_lambda = coef_lambda
        self.Q_star = None
        self.mse_episode = None

    def reset_eligibility(self):
        self.E = np.zeros(self.env_dim)

    def update(self, **params):
        if params['reward'] == 1:
            self.wins += 1

        if "new_obs" in params:
            delta = (
                params['reward']
                + self.g
                * self.Q[self._to_index(params["new_obs"], params["new_act"])]
                - self.Q[self._to_index(params["obs"], params["act"])]
            )
        else:
            delta = (
                params['reward']
                - self.Q[self._to_index(params["obs"], params["act"])]
            )

        self.E[self._to_index(params["obs"], params["act"])] += 1.0
        alpha = 1.0 / self.N[self._to_index(params["obs"], params["act"])]

        self.Q += alpha * delta * self.E
        self.E = self.g * self.coef_lambda * self.E

    def load_q_star(self, filename, n_episodes):
        rl_algo = RLAlgorithm(self.env_dim, 0.0)
        rl_algo.import_data(filename)
        assert self.Q.shape == rl_algo.Q.shape
        self.Q_star = rl_algo.Q

        self.mse_episode = np.zeros(n_episodes)

    def compute_mse(self, episode):
        self.mse_episode[episode] = np.sum(
            np.square(self.Q-self.Q_star))/float(self.Q_star.size)

    def plot_mse(self):
        plt.plot(self.mse_episode)
        plt.ylabel("mse")
        plt.xlabel("episodes")
        plt.show()


class LinearTDLambdaAlgorithm(TDLambdaAlgorithm):
    def __init__(self, env_dim, n_0, coef_lambda, eps=0.05, alpha=0.01):
        super(LinearTDLambdaAlgorithm, self).__init__(env_dim, n_0, coef_lambda)

        self.eps = eps
        self.alpha = alpha

        # Coarse Coding
        self.idx_intervals = {
            "dealer": [(1, 4), (4, 7), (7, 10)],
            "player": [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)],
            "action": [0, 1]
        }
        self.dim_intervals = [len(self.idx_intervals[l])
                              for l in ["dealer", "player", "action"]]
        self.size_intervals = np.prod(self.dim_intervals)
        self.Z = np.zeros(self.size_intervals)  # Linear Eligibility traces
        # self.W = np.zeros(self.size_intervals)
        self.W = np.random.random(self.size_intervals)*0.2

    def reset_traces(self):
        self.Z = np.zeros(self.size_intervals)

    def _to_binary_feature(self, idx):

        phi = np.zeros(self.dim_intervals)

        for d, dl in enumerate(self.idx_intervals["dealer"]):
            dl_check = (dl[0]-1) <= idx[0] <= (dl[1]-1)
            for p, pl in enumerate(self.idx_intervals["player"]):
                pl_check = (pl[0]-1) <= idx[1] <= (pl[1]-1)
                if dl_check and pl_check:
                    phi[(d, p, idx[2])] = 1.0

        log.debug(phi)
        return phi.reshape(self.size_intervals)

    def take_e_greedy_action(self, obs):
        phi = np.zeros((self.act_available.size, self.size_intervals))
        for a in self.act_available:
            phi[a, :] = self._to_binary_feature(self._to_index(obs, a))

        Q = np.dot(phi, self.W)

        if np.random.random() < (1 - self.eps):
            action = np.argmax(Q)
        else:
            action = np.random.choice(self.act_available)

        # Update Q and N (N is not mandatory)
        self.Q[self._to_index(obs, action)] = Q[action]
        self.N[self._to_index(obs, action)] += 1.0

        return action

    def update(self, **params):

        if params['reward'] == 1:
            self.wins += 1

        # Update accumulating traces
        phi = self._to_binary_feature(self._to_index(params["obs"],
                                                     params["act"]))
        self.Z[phi == 1.0] += 1

        delta = (
            params['reward']
            - self.Q[self._to_index(params["obs"], params["act"])]
        )

        if "new_obs" in params:
            delta += (
                self.g
                * self.Q[self._to_index(params["new_obs"], params["new_act"])]
            )

        self.W += self.alpha * delta * self.Z
        self.Z = self.g * self.coef_lambda * self.Z

    def compute_Q(self):
        for dl in np.arange(self.Q.shape[0]):
            for pl in np.arange(self.Q.shape[1]):
                for act in np.arange(self.Q.shape[2]):
                    idx = (dl, pl, act)
                    self.Q[idx] = np.dot(self._to_binary_feature(idx), self.W)


def run_mc_control(n_episodes=10000, n0=100.0):

    easy21 = gym.make('Easy21-v0')

    mc_algo = MonteCarloAlgorithm(easy21.dim, n0)

    for i_episode in range(n_episodes):
        trace = list()
        observation, _, done, info = easy21.reset()
        while not done:
            action = mc_algo.take_e_greedy_action(observation)
            new_observation, reward, done, info = easy21.step(action)
            trace.append((observation, action, reward))
            observation = new_observation

        # Update policy with Episode's experience
        mc_algo.update_policy(trace)

        log.debug("MC Episode: {}".format(i_episode))
        log.debug(trace)
        log.debug(info)

    mc_algo.plot_heatmap()
    mc_algo.plot_surface()
    mc_algo.export_data("mc_pol_{}.plk".format('%.0e' % n_episodes))

    log.info("MC - Episodes {} - Win percentage: {}".format(
        n_episodes, mc_algo.wins/n_episodes * 100.0))
    return mc_algo.wins/n_episodes * 100.0


def run_td_lambda_control(n_episodes=10000, td_lambda=1.0):

    easy21 = gym.make('Easy21-v0')

    tdl_algo = TDLambdaAlgorithm(easy21.dim, 100.0, td_lambda)
    tdl_algo.load_q_star("mc_pol_1e+06.plk", n_episodes)

    for i_episode in range(n_episodes):
        observation, __, done, info = easy21.reset()  # initial state
        tdl_algo.reset_eligibility()
        action = tdl_algo.take_e_greedy_action(observation)
        while not done:
            new_observation, reward, done, info = easy21.step(action)

            update_param = {"reward": reward, "obs": observation,
                            "act": action}

            if not done:  # Terminal state
                new_action = tdl_algo.take_e_greedy_action(new_observation)
                update_param.update({"new_obs": new_observation,
                                     "new_act": new_action})
                action = new_action

            tdl_algo.update(**update_param)

            observation = new_observation

        tdl_algo.compute_mse(i_episode)

    if td_lambda in [0.0, 1.0]:
        tdl_algo.plot_mse()

    # tdl_algo.plot_heatmap()
    # tdl_algo.plot_surface()
    # tdl_algo.export_data("tdl_pol_{}.plk".format("%.0e" % n_episodes))

    log.info("TD({}) - Episodes {} - Win percentage: {}".format(
        td_lambda, n_episodes, tdl_algo.wins/n_episodes * 100.0))


    return tdl_algo.mse_episode[-1]


def run_linear_td_lambda_control(n_episodes=10000, td_lambda=1.0):

    easy21 = gym.make('Easy21-v0')

    ltdl_algo = LinearTDLambdaAlgorithm(easy21.dim, 100.0, td_lambda)
    ltdl_algo.load_q_star("mc_pol_1e+06.plk", n_episodes)

    for i_episode in range(n_episodes):
        observation, __, done, info = easy21.reset()
        ltdl_algo.reset_traces()
        action = ltdl_algo.take_e_greedy_action(observation)

        while not done:
            new_observation, reward, done, info = easy21.step(action)

            update_param = {"reward": reward, "obs": observation,
                            "act": action}

            if not done:
                new_action = ltdl_algo.take_e_greedy_action(new_observation)
                update_param.update({"new_obs": new_observation,
                                     "new_act": new_action})
                action = new_action

            ltdl_algo.update(**update_param)

            observation = new_observation

        ltdl_algo.compute_Q()
        ltdl_algo.compute_mse(i_episode)

    if td_lambda in [0.0, 1.0]:
        ltdl_algo.plot_mse()

    # ltdl_algo.plot_heatmap()
    # ltdl_algo.plot_surface()
    # ltdl_algo.export_data("ltdl_pol_{}.plk".format("%.0e" % n_episodes))

    log.info("Linear TD({}) - Episodes {} - Win percentage: {}".format(
        td_lambda, n_episodes, ltdl_algo.wins/n_episodes * 100.0))

    return ltdl_algo.mse_episode[-1]


def run():
    log.info("Task 2 - Monte-Carlo Control")
    # n_0_coef = np.arange(50, 1050, 50)
    # n_0_wins_pcg = [run_mc_control(10000, n) for n in n_0_coef]
    # plt.plot(n_0_coef, n_0_wins_pcg)
    # plt.show()
    run_mc_control(1000000)

    log.info("Task 3 - SARSA(lambda)")
    lambdas_coef = np.arange(0, 1.1, 0.1)
    lambdas_mse = [run_td_lambda_control(1000, l) for l in lambdas_coef]
    plt.plot(lambdas_coef, lambdas_mse)
    plt.ylabel("mse")
    plt.xlabel("lambda")
    plt.show()

    log.info("Task 4 - SARSA(lambda) - Linear function approximation")
    # run_linear_td_lambda_control(1000, 0.0)
    lambdas_coef = np.arange(0, 1.1, 0.1)
    lambdas_mse = [run_linear_td_lambda_control(1000, l) for l in lambdas_coef]
    plt.plot(lambdas_coef, lambdas_mse)
    plt.ylabel("mse")
    plt.xlabel("lambda")
    plt.show()


if __name__ == '__main__':
    run()