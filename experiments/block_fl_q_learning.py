import numpy as np
import json
from TD_learning.q_learning import QLearningAgent
from environment.block_fl import BlockFLEnv, parameters


TEST_ID = 2
file_name = './results/block_fl_q_learning/q_learning_result_{}.json'.format(TEST_ID)

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
    # 'data_3': [],
    'states': [],
    'actions': [],
    'data_required': [],
    'energy_required': [],
    'latency_required': [],
    'payment_required': [],
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
    nb_devices = 2
    nb_games = 4000
    window = 10

    env = BlockFLEnv(nb_devices=nb_devices, d_max=4, e_max=4, u_max=4, f_max=3, c_max=3, m_max=8)

    parameters['cumulative_data_threshold'] = 1000
    parameters['alpha_D'] = 10
    parameters['alpha_E'] = 3
    parameters['alpha_L'] = 1
    parameters['alpha_I'] = 2
    parameters['training_latency_scale'] = 0.8
    parameters['blk_latency_scale'] = 0.2
    parameters['sigma'] = 1.2 * 10**9
    parameters['penalty_scale'] = 0.5
    parameters['energy_threshold'] = env.nb_devices * (env.e_max - 1)
    parameters['data_threshold'] = env.nb_devices * (env.d_max - 1)
    parameters['payment_threshold'] = parameters['training_price'] * parameters['data_threshold']\
                                       + parameters['blk_price'] / np.log(1 + 1)
    parameters['latency_threshold'] = parameters['transmission_latency'] \
                    + parameters['cross_verify_latency'] \
                    + parameters['block_prop_latency'] \
                    + parameters['blk_latency_scale'] * (env.m_max - 1) \
                    + parameters['training_latency_scale'] * ((parameters['nu']**1.5) * (parameters['tau']**0.5) / parameters['delta']**0.5) * (env.d_max-1)**1.5
                    # 10 ~ max(np.random.exponential(1)) \
    nb_actions = env.d_max ** (2*nb_devices + 1)
    nb_states = (env.f_max + 1) ** (2*nb_devices) * env.m_max

    agent = QLearningAgent(nb_states, nb_actions, alpha=0.01, gamma=0.99,
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
            action = agent.choose_action(s)
            a = process_action(action, env.d_max, env.nb_devices)
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
        json_data['data_required'].append(np.mean(env.logger['data_required']))
        json_data['energy_required'].append(np.mean(env.logger['energy_required']))
        json_data['latency_required'].append(np.mean(env.logger['latency_required']))
        json_data['payment_required'].append(np.mean(env.logger['payment_required']))

        if i >= window and i % window == 0:
            with open(file_name, 'w') as outfile:
                json.dump(json_data, outfile)
            # print(json_data['actions'][i])

        print('Episode: {}, epsilon: {}, steps: {}, scores: {}'.format(i + 1, agent.epsilon, steps, scores))
    print('****************** Q-learning test: {} ends ******************* \n'.format(TEST_ID))



