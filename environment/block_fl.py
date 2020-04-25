"""
Blockchain-enabled federated learning environment
@author: Nguyen Quang Hieu
"""

from __future__ import division
import gym
import numpy as np
from gym.utils import seeding
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box


parameters = {
    'cumulative_data_threshold': 1000,
    'tau': 10**(-28),
    'nu': 10**10,
    'delta': 1,
    'sigma': 0.6 * 10**9,
    'training_price': 0.2,
    'blk_price': 0.8,
    'data_qualities': [1, 1, 1],  # var_1
    'alpha_D': 10,
    'alpha_E': 3,
    'alpha_L': 1,
    'alpha_I': 2,
    'mining_rate_zero': 5,
    'block_arrival_rate': 4,
    'energy_threshold': 9,
    'data_threshold': 9,
    'payment_threshold': 2.955,  # var_1
    'latency_threshold': 54,
    'transmission_latency': 0.0193,  # seconds
    'cross_verify_latency': 0.05,
    'block_prop_latency': 0.01,
    'lambda': 4,
}


class BlockFLEnv(gym.Env):

    def __init__(self, nb_devices=3, d_max=4, e_max=4, u_max=4, f_max=3, c_max=3, m_max=10):
        self.nb_devices = nb_devices
        self.d_max = d_max
        self.e_max = e_max
        self.u_max = u_max
        self.f_max = f_max
        self.c_max = c_max
        self.m_max = m_max

        self.action_space = MultiDiscrete(np.array([self.d_max, self.e_max, self.u_max])
                                          .repeat(nb_devices)[:2*self.nb_devices+1])
        low_box = np.array([0, 0, 1]).repeat(self.nb_devices)[:2*self.nb_devices+1]
        high_box = np.array([self.f_max, self.c_max, self.m_max]).repeat(self.nb_devices)[:2*self.nb_devices+1]
        self.observation_space = Box(low=low_box, high=high_box, dtype=np.int32)
        self.state = self.observation_space.sample()
        self.accumulate_data = np.zeros(nb_devices)

        self.logger = {
            'episode_reward': [],
            'episode_steps': 0,
            'epsilon': 0,
            'average_reward': 0,
            'energy': [],
            'latency': [],
            'payment': [],
            'cumulative_data': np.zeros(self.nb_devices),
            'actions': [],
            'states': []
        }

        self.seed(123)

    def get_penalties(self, data, energy, scale):
        penalties = 0
        for d in data:
            if d == 0:
                penalties += 1
        for e in energy:
            if e == 0:
                penalties += 1
        return penalties * scale

    def check_action(self, action):
        state = self.state
        capacity_array = np.copy(state[self.nb_devices:2*self.nb_devices])
        data_action_array = np.copy(action[0:self.nb_devices])
        energy_action_array = np.copy(action[self.nb_devices:2*self.nb_devices])
        mining_rate_array = np.full(self.nb_devices, parameters['mining_rate_zero'] + action[-1], dtype=int)

        for i in range(len(energy_action_array)):
            if energy_action_array[i] > capacity_array[i]:
                energy_action_array[i] = 0

        corrected_action = np.array([data_action_array, energy_action_array,
                                     mining_rate_array]).flatten()[:2*self.nb_devices+1]
        return corrected_action

    def get_cpu_cycles(self, energy, data):
        cpu_cycles = np.zeros(len(energy))
        cpu_cycles_max = parameters['sigma'] * self.state[:self.nb_devices]
        for i in range(len(data)):
            if data[i] != 0 and energy[i] != 0:
                cpu_cycles[i] = min(np.sqrt(parameters['delta'] * energy[i]
                                            / (parameters['tau'] * parameters['nu'] * data[i])), cpu_cycles_max[i])
            else:
                cpu_cycles[i] = 0
        return cpu_cycles

    def calculate_latency(self, action):
        data = np.copy(action[0:self.nb_devices])
        energy = np.copy(action[self.nb_devices:2 * self.nb_devices])
        mining_rate = parameters['mining_rate_zero'] + action[-1]
        cpu_cycles = self.get_cpu_cycles(energy, data)
        training_latency = np.max([parameters['nu'] * data[k] / cpu_cycles[k] if cpu_cycles[k] != 0 else 0
                                   for k in range(len(data))])
        latency = parameters['transmission_latency'] \
                    + parameters['cross_verify_latency'] \
                    + parameters['block_prop_latency'] \
                    + self.nprandom.exponential(1 / (mining_rate - parameters['block_arrival_rate'])) \
                    + training_latency
        return latency

    def get_reward(self, action):
        data = np.copy(action[0:self.nb_devices])
        energy = np.copy(action[self.nb_devices:2 * self.nb_devices])
        cumulative_data = np.sum([parameters['data_qualities'][k] * data[k] for k in range(self.nb_devices)])
        payment = parameters['training_price'] * cumulative_data + parameters['blk_price'] / np.log(1 + self.state[-1])
        latency = self.calculate_latency(action)
        penalties = self.get_penalties(data, action, 0.8)
        reward = parameters['alpha_D'] * cumulative_data / parameters['data_threshold'] \
                 - parameters['alpha_E'] * np.sum(energy) / parameters['energy_threshold'] \
                 - parameters['alpha_L'] * latency / parameters['latency_threshold'] \
                 - parameters['alpha_I'] * payment / parameters['payment_threshold'] \
                 - penalties

        if latency / parameters['latency_threshold'] > 1:
            print('data: {}, energy: {}, latency: {}, payment: {}'.format(cumulative_data / parameters['data_threshold'],
                                                                        np.sum(energy) / parameters['energy_threshold'],
                                                                        latency / parameters['latency_threshold'],
                                                                         payment / parameters['payment_threshold']))

        self.logger['latency'].append(latency)
        self.logger['energy'].append(np.sum(energy))
        self.logger['payment'].append(payment)
        self.logger['cumulative_data'] = np.add(self.logger['cumulative_data'], data)

        return reward

    def state_transition(self, state, action):
        capacity_array = np.copy(state[self.nb_devices:2*self.nb_devices])
        energy_array = np.copy(action[self.nb_devices:2*self.nb_devices])
        mining_rate = parameters['mining_rate_zero'] + action[-1]
        charging_array = self.nprandom.poisson(1, size=len(energy_array))
        cpu_shares_array = self.nprandom.randint(self.f_max+1, size=self.nb_devices)
        next_capacity_array = np.zeros(len(capacity_array))
        block_queue_state = self.nprandom.geometric(1 - parameters['lambda'] / mining_rate, size=self.nb_devices)
        for i in range(len(next_capacity_array)):
            next_capacity_array[i] = min(capacity_array[i] - energy_array[i] + charging_array[i], self.c_max)
        next_state = np.array([cpu_shares_array, next_capacity_array, block_queue_state], dtype=np.int32).flatten()
        return next_state[:1+2*self.nb_devices]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        corrected_action = self.check_action(action)
        # corrected_action = action
        data = np.copy(corrected_action[0:self.nb_devices])
        state = np.copy(self.state)
        next_state = self.state_transition(state, corrected_action)
        reward = self.get_reward(corrected_action)
        self.accumulate_data = np.add(self.accumulate_data, data)

        self.logger['episode_steps'] += 1
        self.logger['episode_reward'].append(reward)
        self.logger['actions'].append(action)
        self.logger['states'].append(state)

        if np.sum(self.accumulate_data) >= parameters['cumulative_data_threshold']:
            done = True
            self.logger['average_reward'] = np.mean(self.logger['episode_reward'])
        else:
            done = False
        self.state = next_state

        return next_state, reward, done, {}

    def reset(self):
        self.accumulate_data = np.zeros(self.nb_devices)
        self.logger = {
            'episode_reward': [],
            'episode_steps': 0,
            'epsilon': 0,
            'average_reward': 0,
            'energy': [],
            'latency': [],
            'payment': [],
            'cumulative_data': np.zeros(self.nb_devices),
            'actions': [],
            'states': []
        }
        state = self.observation_space.sample()
        return state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]

