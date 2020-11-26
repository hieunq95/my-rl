from environment.block_fl import BlockFLEnv
from dqn.dqn_agent import Agent
import numpy as np
import json


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
    'actions': [],
}


def to_scalar_state(state):
    s = 0
    for i in range(len(state)):
        if i < len(state):
            s += state[i] * 4**(len(state) - i - 1)
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


def train(test_id=1, nb_episodes=2000, eps=1.0, eps_end=0.1, eps_decay=1500, parameters=None):
    env = BlockFLEnv(3, 4, 4, 4, 3, 3, 10, parameters)
    nb_actions = env.d_max ** (2 * env.nb_devices + 1)
    nb_episodes = nb_episodes
    agent = Agent(gamma=0.99, epsilon=eps, epsilon_end=eps_end, alpha=0.001, input_dims=env.observation_space.shape[0],
                  epsilon_dec=(eps - eps_end)/eps_decay, n_actions=nb_actions, mem_size=50000, batch_size=64, replace=1000)
    scores = []
    print(env.action_space, env.observation_space, nb_actions)
    print(parameters)
    print('\n ****************** DQN test: {} begins ******************* \n'.format(test_id))
    for i in range(nb_episodes):
        done = False
        score = 0
        step = 0
        rewards = []
        observation = env.reset()
        while not done:
            step += 1
            action = agent.choose_action(observation)
            processed_action = process_action(action, env.d_max, env.nb_devices)
            observation_, reward, done, info = env.step(processed_action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        agent.update_epsilon()
        scores.append(score)

        print('episode: %4d, epsilon: %5.2f, steps: %4d, reward: %7.2f, average_reward: %5.2f,'
              ' energy: %5.2f, latency: %5.2f, payment: %5.2f'
              %(i + 1, agent.epsilon, env.logger['episode_steps'],
                      np.sum(env.logger['episode_reward']), env.logger['average_reward'],
                np.mean(env.logger['energy']), np.mean(env.logger['latency']), np.mean(env.logger['payment'])))
        json_data['episode'].append(i + 1)
        json_data['reward'].append(np.sum(env.logger['episode_reward']))
        json_data['avg_reward'].append(env.logger['average_reward'])
        json_data['energy'].append(np.mean(env.logger['energy']))
        json_data['latency'].append(np.mean(env.logger['latency']))
        json_data['payment'].append(np.mean(env.logger['payment']))
        json_data['epsilon'].append(agent.epsilon)
        json_data['data_1'].append(env.logger['cumulative_data'][0])
        json_data['data_2'].append(env.logger['cumulative_data'][1])
        json_data['data_3'].append(env.logger['cumulative_data'][2])
        json_data['states'].append(np.mean([to_scalar_state(s) for s in env.logger['states']]))
        json_data['actions'].append(np.mean([to_scalar_action(a, env.d_max) for a in env.logger['actions']]))

        with open('./results/block_fl/result_{}.json'.format(test_id), 'w') as outfile:
            json.dump(json_data, outfile)
    print('****************** DQN test: {} ends ******************* \n'.format(TEST_ID))
    model_path = './results/block_fl/block_fl_dqn.h5'
    print('Save model to: {}'.format(model_path))
    agent.save_model(model_path)


if __name__ == '__main__':
    parameters = {
        'cumulative_data_threshold': 1000,
        'tau': 10 ** (-28),
        'nu': 10 ** 10,
        'delta': 1,
        'sigma': 0.6 * 10 ** 9,
        'training_price': 0.2,
        'blk_price': 0.8,
        'data_qualities': [1, 1, 1],  # var_1
        'alpha_D': 10,
        'alpha_E': 1,
        'alpha_L': 3,
        'alpha_I': 2,
        'mining_rate_zero': 5,  # 5 blocks/hour
        'block_arrival_rate': 4,
        'energy_threshold': 9,
        'data_threshold': 9,
        'payment_threshold': 2.955,  # var_1 - 3 devices, 0.2 * D + 0.8 / log(1+m) = 2.955
        'latency_threshold': 540,
        'transmission_latency': 0.0193,  # seconds
        'cross_verify_latency': 0.05,
        'block_prop_latency': 0.01,
        'lambda': 4,
        'training_latency_scale': 1,
        'blk_latency_scale': 60,  # minutes
        'penalty_scale': 1,
    }
    train(test_id=1, nb_episodes=2000, eps=1.0, eps_end=0.1, eps_decay=1500, parameters=parameters)
