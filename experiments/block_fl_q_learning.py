import numpy as np
import json
from TD_learning.q_learning import QLearningAgent
from environment.block_fl import BlockFLEnv, parameters


TEST_ID = 2

json_data = {
    'epsilon': [],
    'episode': [],
    'reward': [],
    'avg_reward': [],
    'energy': [],
    'latency': [],
    'payment': [],
    'data_1': [],
    'data_2': [],
    'data_3': [],
    'states': [],
    'actions': []
}


def to_scalar_state(state, order):
    s = 0
    for i in range(len(state)):
        if i < len(state):
            s += state[i] * order**(len(state) - i - 1)
        else:
            s += state[i]
    return s


def to_scalar_action(action, order):
    a = 0
    for i in range(len(action)):
        a += action[i] * order**(len(action) - i - 1)
    return a


def process_action(action, action_limit, nb_devices):
    """
    Convert action from decima number to array
    :param action: a number
    :param the maximum number of an action in an array
    :return: an array
    """
    data_array, energy_array, payment_array = [], [], []
    action_size = 2*nb_devices+1
    for i in range(action_size):
        if i < nb_devices:
            divisor = action_limit ** (action_size - (i+1))
            data_i = action // divisor
            action -= data_i * divisor
            data_array.append(data_i)
        elif i >= nb_devices and i < 2*nb_devices:
            divisor = action_limit ** (action_size - (i+1))
            energy_i = action // divisor
            action -= energy_i * divisor
            energy_array.append(energy_i)
        elif i >= 2*nb_devices:
            divisor = action_limit ** (action_size - (i+1))
            payment_i = action // divisor
            action -= payment_i * divisor
            payment_array.append(payment_i)

    payment = np.full(nb_devices, payment_array[0])
    processed_action = np.array([data_array, energy_array, payment]).flatten()[:2*nb_devices+1]
    return processed_action


if __name__ == '__main__':
    nb_devices = 3
    nb_games = 4000
    window = 10

    parameters['cumulative_data_threshold'] = 1000
    parameters['alpha_D'] = 10
    parameters['alpha_E'] = 3
    parameters['alpha_L'] = 1
    parameters['alpha_I'] = 2

    env = BlockFLEnv(nb_devices=nb_devices, d_max=4, e_max=4, u_max=4, f_max=3, c_max=3, m_max=10)

    nb_actions = env.d_max ** (2*nb_devices + 1)
    nb_states = (env.f_max + 1) ** (2*nb_devices) * env.m_max

    agent = QLearningAgent(nb_states, nb_actions, alpha=0.001, gamma=0.99,
                           epsilon=0.9, epsilon_min=0.1, epsilon_decay=4e-4)


    print(env.action_space, env.observation_space.sample(), env.observation_space.low, env.observation_space.high)
    print(agent.q_table.shape)
    print(parameters)
    print('****************** Q-learning test: {} begins ******************* \n'.format(TEST_ID))

    for i in range(nb_games):
        done = False
        state = env.reset()
        scores = 0
        steps = 0

        while not done:
            s = to_scalar_state(state, env.f_max+1)
            random_action = env.action_space.sample()
            action = agent.choose_action(s)
            a = process_action(action, env.d_max, nb_devices)
            next_state, reward, done, _ = env.step(a)
            s_ = to_scalar_state(next_state, env.f_max+1)
            agent.update_q_table(reward, a, s, s_, done)
            state = next_state

            scores += reward
            steps += 1
        agent.epsilon_update()
        # print('max q_table: {}'.format(agent.q_table[to_scalar_state(state)]))

        json_data['episode'].append(i + 1)
        json_data['reward'].append(np.sum(env.logger['episode_reward']))
        json_data['avg_reward'].append(env.logger['average_reward'])
        json_data['energy'].append(np.mean(env.logger['energy']))
        json_data['latency'].append(np.mean(env.logger['latency']))
        json_data['payment'].append(np.mean(env.logger['payment']))
        json_data['epsilon'].append(agent.epsilon)
        json_data['data_1'].append(env.logger['cumulative_data'][0])
        json_data['data_2'].append(env.logger['cumulative_data'][1])
        if env.nb_devices > 2:
            json_data['data_3'].append(env.logger['cumulative_data'][2])
        json_data['states'].append(np.mean([to_scalar_state(s, 3) for s in env.logger['states']]))
        json_data['actions'].append(np.mean([to_scalar_action(a, 3) for a in env.logger['actions']]))

        if i >= window and i % window == 0:
            with open('./results/q_learning_result_{}.json'.format(TEST_ID), 'w') as outfile:
                json.dump(json_data, outfile)
            # print(json_data['actions'][i])

        print('Episode: {}, epsilon: {}, steps: {}, scores: {}'.format(i + 1, agent.epsilon, steps, scores))
    print('****************** Q-learning test: {} ends ******************* \n'.format(TEST_ID))



