import numpy as np


class NstepsSarsa(object):
    def __init__(self, nb_states, nb_actions, epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-4,
                 alpha=0.1, gamma=0.99, n=4, mem_size=100):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.action_space = [i for i in range(self.nb_actions)]
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.mem_size = mem_size
        self.q_table = np.zeros((self.nb_states, self.nb_actions))
        self.memory = np.zeros((3, mem_size), dtype=int) #  memory = [state: ...
                                                            # action: ...,
                                                            # reward: ...,
                                                            # ]
        self.nsteps_return = 0

    def update_q_table(self, state, action, nsteps_return):
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.alpha*(nsteps_return - self.q_table[state, action])

    def update_nsteps_return(self, state, action):
        self.nsteps_return = self.nsteps_return + (self.gamma**self.n)*self.q_table[state, action]

    def set_nsteps_return(self, tau, T):
        low = tau + 1
        high = min(tau + self.n, T)
        self.nsteps_return = np.sum([self.gamma**(i - tau - 1) * self.memory[2, i] for i in range(low, high + 1)])
        return self.nsteps_return

    def store_in_memory(self, value, index, type):
        '''
        Store s_t, a_t, r_t in the memory
        :param value: value to be stored
        :param index: index in the array of memory
        :param type: 0-state, 1-action, 2-reward
        :return: memory
        '''
        # type in [0, 1, 2]
        self.memory[type, index] = value

    def choose_action(self, state):
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay \
            if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min
