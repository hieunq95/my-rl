import numpy as np
import random


class QLearningAgent(object):
    def __init__(self, nb_states, nb_actions, alpha=0.01, gamma=0.99,
                         epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-4):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.action_space = np.arange(self.nb_actions)
        self.q_table = np.zeros((self.nb_states, self.nb_actions))

    def choose_action(self, state):
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def update_q_table(self, reward, action, state,  next_state, done):
        old_q = self.q_table[state, action]
        if not done:
            new_q = old_q + self.alpha*(reward + self.gamma*np.max(self.q_table[next_state]) - old_q)
        else:
            new_q = old_q
        self.q_table[state, action] = new_q

    def epsilon_update(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

