import json
import pandas
import matplotlib.pyplot as plt
import numpy as np


"""
result_1
nsteps_sarsa_result_1
q_learning_result_1 nsteps_sarsa_result_cloud.json
"""
f1 = './results/result_2_cloud.json'
f2 = './results/nsteps_sarsa_result_1.json'
f3 = './results/block_fl_q_learning/q_learning_result_2.json'
f4 = './results/block_fl_random/random_agent.json'
f = f3

evaluated_metric = 'scores'
window = 10
metrics = ['reward', 'avg_reward', 'data_required', 'energy_required', 'latency_required', 'payment_required',
           'energy', 'latency', 'payment', 'data_1', 'data_2', 'actions', 'states']


def plot_learning(file, ewm_window, metric):
    x_array, y_array, epsilons = [], [], []
    with open(file) as json_file:
        data = json.load(json_file)
        ewm_data = pandas.read_json(f)[metric].ewm(span=ewm_window, adjust=False).mean()
        i = 0
        for p in data[metric]:
            x_array.append(i)
            y_array.append(ewm_data[i])
            i += 1
        plt.plot(x_array, y_array, label=metric)
        print('{}: {}'.format(metric, np.mean(y_array[-100:])))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    for i in range(len(metrics)):
        plot_learning(f, window, metrics[i])
    # plot_learning(f, window, evaluated_metric)
