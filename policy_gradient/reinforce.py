"""
Implementation of REINFORCE algorithm
"""
from __future__ import division
import numpy as np
import json
from environment.short_corridor import ShortCorridor


class REINFORCE(object):
    def __init__(self, nb_actions, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        # self.theta = np.zeros((2, ))
        self.theta = np.array([-1.47, 1.47])
        self.x = np.array([[0, 1],
                           [1, 0]])
        self.action_space = [i for i in range(nb_actions)]
        self.gt = 0
        print(self.x)

    def return_value_update(self, t, T, reward):
        self.gt = np.sum([reward*self.gamma**(k - t - 1) for k in range(t+1, T+1)])
        return self.gt

    def policy_parameter_update(self, t, T, reward, action, state):
        self.theta = self.theta + self.alpha*self.gamma**t * self.gt \
                     * self.get_gradient(state, action)
        return self.theta

    def choose_action(self, state):
        policy = self.get_pi(state)
        # print(policy)
        if np.random.uniform() <= policy[1]:
            return 1
        else:
            return 0

    def get_gradient(self, state, action):
        sum1 = np.sum([self.x[:, a]*np.exp(self.get_h(a)) for a in self.action_space])
        sum2 = np.sum([np.exp(self.get_h(a)) for a in self.action_space])
        gradient = self.x[:, action] - sum1 / sum2
        return gradient

    def get_h(self, action):
        return self.theta.T.dot(self.x[:, action])

    def get_pi(self, state):
        h = [self.get_h(a) for a in self.action_space]
        pi = np.exp(h) / np.sum([np.exp(self.get_h(a)) for a in self.action_space])
        imin = np.argmin(pi)
        epsilon = 0.05
        if np.min(pi) < epsilon:
            pi[:] = 1 - epsilon
            pi[imin] = epsilon

        return pi


if __name__ == '__main__':

    logger = {
        'scores': [],
        'episode': [],
    }
    file_name = '../experiments/results/policy_gradient/reinforce.json'
    env = ShortCorridor()
    agent = REINFORCE(nb_actions=2, alpha=2**(-12), gamma=0.99)

    for i in range(1000):
        scores = 0
        done = False
        states = []
        actions = []
        rewards = []
        s = env.reset()

        while not done:
            states.append(s)
            a = agent.choose_action(s)
            actions.append(a)
            s_, r, done, _ = env.step(a)
            rewards.append(r)
            s = s_
        scores = int(np.sum(rewards))
        T = len(rewards)
        # print(agent.theta)
        for t in range(T):
            agent.return_value_update(t, T, rewards[t])
            agent.policy_parameter_update(t, T, rewards[t], actions[t], states[t])

        logger['episode'].append(i + 1)
        logger['scores'].append(scores)

        if i >= 10 and i % 10 == 0:
            with open(file_name, 'w') as outfile:
                json.dump(logger, outfile)

        print('Episode: {}, scores: {}'.format(i, scores))
