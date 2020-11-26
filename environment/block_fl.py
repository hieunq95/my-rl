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


local_parameters = {
    'cumulative_data_threshold': 1000,
    'tau': 10**(-28),
    'nu': 10**10,
    'delta': 1,
    'sigma': 0.6 * 10**9,
    'training_price': 0.2,
    'blk_price': 0.8,
    'data_qualities': [1, 1, 1],  # var_1
    'alpha_D': 10,
    'alpha_E': 1,
    'alpha_L': 1,
    'alpha_I': 1,
    'mining_rate_zero': 5,  # 5 blocks/hour
    'block_arrival_rate': 4,
    'energy_threshold': 9,
    'data_threshold': 9,
    'payment_threshold': 4.458,  # var_1
    'latency_threshold': 54,
    'transmission_latency': 0.0193,  # seconds
    'cross_verify_latency': 0.05,
    'block_prop_latency': 0.01,
    'lambda': 4,
    'training_latency_scale': 1,
    'blk_latency_scale': 60,  # minutes
    'penalty_scale': 1,
}

queue_latency_max = np.max(np.random.exponential(1 / (0 + local_parameters['mining_rate_zero']
                                                      - local_parameters['block_arrival_rate']), 1000000))
local_parameters['latency_threshold'] = 51.9612 + local_parameters['transmission_latency'] \
                                  + local_parameters['blk_latency_scale'] * queue_latency_max + \
                                  local_parameters['block_prop_latency'] + local_parameters['cross_verify_latency']

print('latency_threshold: {}'.format(local_parameters['latency_threshold']))


class BlockFLEnv(gym.Env):

    def __init__(self, nb_devices=3, d_max=4, e_max=4, u_max=4, f_max=3, c_max=3, m_max=10, parameters=None):
        self.parameters = parameters
        self.parameters['latency_threshold'] = local_parameters['latency_threshold']
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
        # self.state = self.observation_space.sample()
        self.accumulate_data = np.zeros(self.nb_devices)
        self.penalties = 0

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
            'states': [],
            'data_required': [],
            'energy_required': [],
            'latency_required': [],
            'payment_required': [],
        }
        self.seed(123)
        self.reset()

    def get_penalties(self, scale):

        return self.penalties * scale

    def check_action(self, action):
        self.penalties = 0
        state = np.copy(self.state)
        capacity_array = np.copy(state[self.nb_devices:2*self.nb_devices])
        data_action_array = np.copy(action[0:self.nb_devices])
        energy_action_array = np.copy(action[self.nb_devices:2*self.nb_devices])
        mining_rate_array = np.full(self.nb_devices, self.parameters['mining_rate_zero'] + action[-1], dtype=int)
        cpu_cycles = self.get_cpu_cycles(energy_action_array, data_action_array)

        for i in range(len(energy_action_array)):
            if energy_action_array[i] > capacity_array[i]:
                # energy_action_array[i] = capacity_array[i]
                energy_action_array[i] = 0
                self.penalties += 0

        for j in range(len(cpu_cycles)):
            if cpu_cycles[j] == 0:
                data_action_array[j] = 0
                energy_action_array[j] = 0
                self.penalties += 0

        corrected_action = np.array([data_action_array, energy_action_array,
                                     mining_rate_array]).flatten()[:2*self.nb_devices+1]
        return corrected_action

    def get_cpu_cycles(self, energy, data):
        cpu_cycles = np.zeros(len(energy))
        cpu_cycles_max = self.parameters['sigma'] * self.state[:self.nb_devices]
        for i in range(len(data)):
            if data[i] != 0 and energy[i] != 0:
                cpu_cycles[i] = np.sqrt(self.parameters['delta'] * energy[i]
                                        / (self.parameters['tau'] * self.parameters['nu'] * data[i]))
                if cpu_cycles[i] > cpu_cycles_max[i]:
                    cpu_cycles[i] = 0
            else:
                cpu_cycles[i] = 0
        # print(cpu_cycles)
        return cpu_cycles

    def calculate_latency(self, action):
        data = np.copy(action[:self.nb_devices])
        energy = np.copy(action[self.nb_devices:2 * self.nb_devices])
        mining_rate = self.parameters['mining_rate_zero'] + action[-1]
        cpu_cycles = self.get_cpu_cycles(energy, data)
        training_latency = np.max([self.parameters['nu'] * data[k] / cpu_cycles[k] if cpu_cycles[k] != 0 else 0
                                   for k in range(len(data))])
        block_queue_latency = self.parameters['cross_verify_latency'] + self.parameters['block_prop_latency'] + \
                              self.parameters['blk_latency_scale'] * self.nprandom.exponential(1 / (mining_rate - self.parameters['block_arrival_rate']))
        latency = self.parameters['transmission_latency'] + block_queue_latency +\
                  self.parameters['training_latency_scale'] * training_latency
        # print('L_tr: {}, L_tx: {}, L_blk: {}'.format(training_latency, parameters['transmission_latency'],
        #                                              block_queue_latency))
        return latency

    def get_reward(self, action):
        data = np.copy(action[:self.nb_devices])
        energy = np.copy(action[self.nb_devices:2 * self.nb_devices])
        cumulative_data = np.sum([self.parameters['data_qualities'][k] * data[k] for k in range(self.nb_devices)])
        payment = self.parameters['training_price'] * cumulative_data + self.parameters['blk_price'] / np.log(1 + self.state[-1])
        latency = self.calculate_latency(action)
        penalties = self.get_penalties(self.parameters['penalty_scale'])
        reward = self.parameters['alpha_D'] * cumulative_data / self.parameters['data_threshold'] \
                 - self.parameters['alpha_E'] * np.sum(energy) / self.parameters['energy_threshold'] \
                 - self.parameters['alpha_L'] * latency / self.parameters['latency_threshold'] \
                 - self.parameters['alpha_I'] * payment / self.parameters['payment_threshold'] \
                 - penalties

        if payment / self.parameters['payment_threshold'] > 1:
            print('data: {}, energy: {}, latency: {}, payment: {}'.format(cumulative_data / self.parameters['data_threshold'],
                                                                        np.sum(energy) / self.parameters['energy_threshold'],
                                                                        latency / self.parameters['latency_threshold'],
                                                                         payment / self.parameters['payment_threshold']))

        self.logger['latency'].append(latency)
        self.logger['energy'].append(np.sum(energy))
        self.logger['payment'].append(payment)
        self.logger['cumulative_data'] = np.add(self.logger['cumulative_data'], data)

        return reward, cumulative_data / self.parameters['data_threshold'], np.sum(energy) / self.parameters['energy_threshold'],\
                    latency / self.parameters['latency_threshold'], payment / self.parameters['payment_threshold']

    def state_transition(self, state, action):
        capacity_array = np.copy(state[self.nb_devices:2*self.nb_devices])
        energy_array = np.copy(action[self.nb_devices:2*self.nb_devices])
        mining_rate = self.parameters['mining_rate_zero'] + action[-1]
        charging_array = self.nprandom.poisson(1, size=len(energy_array))
        cpu_shares_array = self.nprandom.randint(low=0, high=self.f_max+1, size=self.nb_devices)
        next_capacity_array = np.zeros(len(capacity_array))
        block_queue_state = self.nprandom.geometric(1 - self.parameters['lambda'] / mining_rate, size=self.nb_devices)
        for i in range(len(next_capacity_array)):
            next_capacity_array[i] = min(capacity_array[i] - energy_array[i] + charging_array[i], self.c_max)
        next_state = np.array([cpu_shares_array, next_capacity_array, block_queue_state], dtype=np.int32).flatten()
        self.state = next_state[:1+2*self.nb_devices]
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        corrected_action = self.check_action(action)
        # corrected_action = action
        data = np.copy(corrected_action[:self.nb_devices])
        state = np.copy(self.state)
        next_state = self.state_transition(state, corrected_action)
        reward = self.get_reward(corrected_action)
        self.accumulate_data = np.add(self.accumulate_data, data)
        # print(self.get_reward(corrected_action)[1:])

        self.logger['episode_steps'] += 1
        self.logger['episode_reward'].append(reward[0])
        self.logger['actions'].append(action)
        self.logger['states'].append(state)
        self.logger['data_required'].append(reward[1])
        self.logger['energy_required'].append(reward[2])
        self.logger['latency_required'].append(reward[3])
        self.logger['payment_required'].append(reward[4])

        if np.sum(self.accumulate_data) >= self.parameters['cumulative_data_threshold']:
            done = True
            self.logger['average_reward'] = np.mean(self.logger['episode_reward'])
        else:
            done = False
        # self.state = next_state

        return next_state, reward[0], done, {}

    def reset(self):
        self.accumulate_data = np.zeros(self.nb_devices)
        self.penalties = 0
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
            'states': [],
            'data_required': [],
            'energy_required': [],
            'latency_required': [],
            'payment_required': [],
        }
        cpu_shares_init = self.nprandom.randint(self.f_max + 1, size=self.nb_devices)
        capacity_init = self.nprandom.randint(self.c_max + 1, size=self.nb_devices)
        mempool_init = np.full(self.nb_devices, 1)
        state = np.array([cpu_shares_init, capacity_init, mempool_init]).flatten()
        state = state[:2 * self.nb_devices + 1]
        # state = self.observation_space.sample()
        self.state = state
        return state

    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]

