import numpy as np
import json
from n_steps_bootstrapping.n_steps_sarsa import NstepsSarsa
from environment.block_fl import BlockFLEnv, parameters


TEST_ID = 1
file_name = './results/nsteps_sarsa_result_{}.json'.format(TEST_ID)

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
    nb_episodes = 20000
    window = 10
    parameters['cumulative_data_threshold'] = 1000

    env = BlockFLEnv(nb_devices=nb_devices, d_max=3, e_max=3, u_max=3, f_max=2, c_max=2, m_max=6)

    nb_actions = env.d_max ** (2 * nb_devices + 1)
    nb_states = (env.f_max + 1) ** (2 * nb_devices) * env.m_max

    agent = NstepsSarsa(nb_states=nb_states, nb_actions=nb_actions,
                        epsilon=0.9, epsilon_min=0.1, epsilon_decay=8e-5,
                        alpha=0.01, gamma=0.99, n=10, mem_size=100000)
    print(agent.q_table.shape)
    print(parameters)
    print('\n******************* n-steps_Sarsa test: {} begins ********************** \n'.format(TEST_ID))

    for i in range(nb_episodes):
        s0 = to_scalar_state(env.reset(), env.f_max+1)
        agent.store_in_memory(s0, 0, 0)
        a0 = agent.choose_action(s0)
        agent.store_in_memory(a0, 0, 1)
        T = agent.mem_size
        tau = 0

        scores = 0
        steps = 0
        for t in range(T):
            if t < T:
                at = agent.memory[1, t]
                st_, rt_, done, _ = env.step(process_action(at, env.d_max, nb_devices))
                st_ = to_scalar_state(st_, env.f_max+1)
                agent.store_in_memory(st_, t+1, 0)
                agent.store_in_memory(rt_, t+1, 2)

                scores += rt_
                steps += 1
                if done:
                    T = t + 1
                else:
                    at_ = agent.choose_action(st_)
                    agent.store_in_memory(at_, t+1, 1)
            tau = t - agent.n + 1
            if tau >= 0:
                agent.set_nsteps_return(tau, T)
                if tau + agent.n < T:
                    s_tn = agent.memory[0, tau + agent.n]
                    a_tn = agent.memory[1, tau + agent.n]
                    agent.update_nsteps_return(s_tn, a_tn)
                    s_t = agent.memory[0, tau]
                    a_t = agent.memory[1, tau]
                    agent.update_q_table(s_t, a_t, agent.nsteps_return)

            if tau == T - 1:
                break
        agent.update_epsilon()

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
        json_data['states'].append(np.mean([to_scalar_state(s, env.f_max+1) for s in env.logger['states']]))
        json_data['actions'].append(np.mean([to_scalar_action(a, env.d_max) for a in env.logger['actions']]))

        if i >= window and i % window == 0:
            with open(file_name, 'w') as outfile:
                json.dump(json_data, outfile)
        print('Episode: {}, epsilon: {}, steps: {}, scores: {}'.format(i + 1, agent.epsilon, steps, scores))
    print('****************** n-steps Sarsa test: {} ends ******************* \n'.format(TEST_ID))

