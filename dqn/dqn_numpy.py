import numpy as np
import gym
from dqn.neural_net import Network


class ReplayMemory(object):
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_idx = 0
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_idx % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward

        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.mem_idx += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-5, epsilon_end=0.01,
                 mem_size=10000, replace=10000):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.learn_step = 0
        self.replace = replace
        self.memory = ReplayMemory(mem_size, input_dims, n_actions)
        self.q_policy = Network([input_dims, 32, 32, n_actions])
        self.q_target = Network([input_dims, 32, 32, n_actions])

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            print(state)
            actions = self.q_policy.feedforward(state)
            action = np.argmax(actions)
        return action

    def replace_target_network(self):
        if self.replace != 0 and self.learn_step % self.replace == 0:
            w, b = self.q_policy.save_weights()
            self.q_target.load_weights(w, b)

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        self.replace_target_network()

        actions_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, actions_values)

        q_eval = self.q_policy.feedforward(state)
        q_next = self.q_target.feedforward(next_state)
        # print(q_eval, q_next)

        q_target = q_eval[:]

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

        self.q_policy.backprop(state, q_target)

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec \
            if self.epsilon - self.epsilon_dec > self.epsilon_end else self.epsilon_end


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    input_dims = env.observation_space.shape[0]
    print(input_dims)
    n_actions = env.action_space.n

    scores =[]

    agent = Agent(0.001, 0.99, n_actions, 0.9, 32, input_dims, 0.008,
                  0.01, 10000, 1000)

    for i in range(1000):
        state = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

            steps += 1
            score += reward

        print('Episode: {}, epsilon: {}, steps: {}, scores: {}'
              .format(i+1, agent.epsilon, steps, score))



