from __future__ import division
import numpy as np
from TD_linear_function_approximation.tile_coding import tile_encode


class SemiGradientSarsa(object):
    def __init__(self, nb_states, nb_actions, w_dims, alpha, gamma, epsilon, tilings, epsilon_end, epsilon_decay):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.action_space = [i for i in range(self.nb_actions)]
        self.w_dims = w_dims
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tilings = tilings
        self.weights_vector = np.zeros(self.w_dims, dtype=float)
        self.feature_vector = np.zeros(self.w_dims, dtype=int)

    def choose_action(self, state):
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = [self.get_q_value(state, a) for a in self.action_space]
            action = np.argmax(actions)
        return action

    def weights_update(self, reward, state, action, next_state, next_action, done):
        temporal_error = reward \
                         + self.gamma*self.get_q_value(next_state, next_action)*(1 - int(done)) \
                         - self.get_q_value(state, action)
        feature_vector = tile_encode(state, action, self.tilings, True)
        self.weights_vector = np.add(self.weights_vector, self.alpha*temporal_error*feature_vector)

        return self.weights_vector

    def get_q_value(self, state, action):
        feature_vector = tile_encode(state, action, self.tilings, True)
        q_value = self.weights_vector.T.dot(feature_vector)
        return q_value

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay \
            if self.epsilon - self.epsilon_decay > self.epsilon_end else self.epsilon_end




