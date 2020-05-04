import numpy as np
import json
from environment.block_fl import BlockFLEnv, parameters

json_data = {
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


file_name = './results/block_fl_random/random_agent.json'


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


if __name__ == '__main__':
    nb_games = 40000
    window = 100
    parameters['cumulative_data_threshold'] = 1000
    parameters['alpha_D'] = 10
    parameters['alpha_E'] = 3
    parameters['alpha_L'] = 1
    parameters['alpha_I'] = 2

    env = BlockFLEnv(nb_devices=3, d_max=4, e_max=4, u_max=4, f_max=3, c_max=3, m_max=8)

    nb_actions = env.d_max ** (2 * env.nb_devices + 1)
    nb_states = (env.f_max + 1) ** (2 * env.nb_devices) * env.m_max

    print(env.action_space, env.observation_space.sample(), env.observation_space.low, env.observation_space.high)
    print(nb_states, nb_actions)
    print(parameters)
    print('****************** Random-agent test - begins ******************* \n')
    exp = np.random.exponential(1, 100000)
    print('max_exp: {}\n'.format(np.max(exp)))
    for i in range(nb_games):
        state = env.reset()
        done = False
        scores = 0
        steps = 0

        while not done:
            env.seed(1000)
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            scores += reward
            steps += 1
        if i % window == 0:
            print('Episode: {}, steps: {}, reward: {}'.format(i, steps, scores))

        json_data['episode'].append(i + 1)
        json_data['reward'].append(np.sum(env.logger['episode_reward']))
        json_data['avg_reward'].append(env.logger['average_reward'])
        json_data['energy'].append(np.mean(env.logger['energy']))
        json_data['latency'].append(np.mean(env.logger['latency']))
        json_data['payment'].append(np.mean(env.logger['payment']))
        json_data['data_1'].append(env.logger['cumulative_data'][0])
        json_data['data_2'].append(env.logger['cumulative_data'][1])
        if env.nb_devices > 2:
            json_data['data_3'].append(env.logger['cumulative_data'][2])
        json_data['states'].append(np.mean([to_scalar_state(s, 3) for s in env.logger['states']]))
        json_data['actions'].append(np.mean([to_scalar_action(a, 3) for a in env.logger['actions']]))

        if i > 0 and i % window == 0:
            with open(file_name, 'w') as outfile:
                json.dump(json_data, outfile)

    print('****************** Random-agent test - ends ******************* \n')