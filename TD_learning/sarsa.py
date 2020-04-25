import numpy as np


class SARSA(object):
    def __init__(self, nb_states, nb_actions, alpha=0.01, gamma=0.99,
                 epsilon1=0.1, epsilon_min1=0.1, epsilon_decay1=1e-4,
                 epsilon2=1.0, epsilon_min2=0.1, epsilon_decay2=1e-4):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon1 = epsilon1
        self.epsilon_min1 = epsilon_min1
        self.epsilon_decay1 = epsilon_decay1
        self.epsilon2 = epsilon2
        self.epsilon_min2 = epsilon_min2
        self.epsilon_decay2 = epsilon_decay2
        self.action_space = [i for i in range(self.nb_actions)]
        self.q_table = np.zeros((self.nb_states, self.nb_actions))

    def choose_action(self, state, next_max=False):
        rand1 = np.random.uniform(0, 1)
        rand2 = np.random.uniform(0, 1)
        rand = rand2 if next_max else rand1
        epsilon = self.epsilon2 if next_max else self.epsilon1
        if rand < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state, :])

        return action

    def update_q_table(self, reward, action, state, next_action, next_state, done):
        old_q = self.q_table[state, action]
        if not done:
            new_q = old_q + self.alpha*(reward + self.gamma*self.q_table[next_state, next_action] - old_q)
        else:
            new_q = 0
        self.q_table[state, action] = new_q

    def epsilon_update(self, next_max=False):
        if not next_max:
            self.epsilon1 = self.epsilon1 -  self.epsilon_decay1 \
                if self.epsilon1 > self.epsilon_min1 else self.epsilon_min1
        else:
            self.epsilon2 = self.epsilon2 - self.epsilon_decay2 \
                if self.epsilon2 > self.epsilon_min2 else self.epsilon_min2



