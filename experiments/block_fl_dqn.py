from environment.block_fl import BlockFLEnv, parameters
from dqn.dqn_agent import Agent
import numpy as np
import json

TEST_ID = 3

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


if __name__ == '__main__':
    env = BlockFLEnv(3, 3, 3, 3, 2, 2, 6)
    nb_actions = env.d_max ** (2 * env.nb_devices + 1)
    n_games = 2000
    agent = Agent(gamma=0.99, epsilon=0.1, epsilon_end=0.1, alpha=0.001, input_dims=7, epsilon_dec=1e-2,
                  n_actions=nb_actions, mem_size=50000, batch_size=64, replace=1000)

    # agent.load_model()
    scores = []

    print(env.action_space, env.observation_space)
    print(parameters)
    print('\n ****************** DQN test: {} begins ******************* \n'.format(TEST_ID))
    for i in range(n_games):
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

        avg_score = np.mean(scores)
        print('episode: {}, epsilon: {}, steps: {}, score: {}, average_score: {}'
              .format(i+1, agent.epsilon, env.logger['episode_steps'],
                      np.sum(env.logger['episode_reward']), env.logger['average_reward']))
        json_data['episode'].append(i+1)
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

        with open('./results/result_{}.json'.format(TEST_ID), 'w') as outfile:
            json.dump(json_data, outfile)
    print('****************** DQN test: {} ends ******************* \n'.format(TEST_ID))
        # if i % 10 == 0 and i > 0:
        #     agent.save_model()
    filename = 'dqn_lunarlander.h5'
