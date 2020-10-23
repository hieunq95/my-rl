import json
import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
import colorsys

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # colors used in plotting figures later


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_learning(file, ewm_window, metric):
    x_array, y_array, epsilons = [], [], []
    y_bound = []
    with open(file) as json_file:
        data = pandas.read_json(json_file)
        ewm_data = data[metric].ewm(span=ewm_window, adjust=False).mean()
        i = 0
        for p in data[metric]:
            x_array.append(i)
            y_array.append(ewm_data[i])
            y_bound.append(p)
            i += 1

        plt.plot(x_array, y_bound, label=evaluated_metric, color=adjust_lightness(color_cycle[0], 1.9))
        plt.plot(x_array, y_array, label='mean-' + evaluated_metric, color=color_cycle[0])
        print('{}: {}'.format(metric, np.mean(y_array[-100:])))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """
    result_1
    nsteps_sarsa_result_1
    q_learning_result_1 nsteps_sarsa_result_cloud.json
    """
    f1 = './results/result_3.json'
    f2 = './results/nsteps_sarsa_result_1.json'
    f3 = './results/block_fl_q_learning/q_learning_result_2.json'
    f4 = './results/block_fl_random/random_agent.json'
    f5 = './results/mountain_car_semi_gradient_sarsa/result.json'
    f6 = './results/mountain_car_q_learning/result_1.json'
    f7 = './results/mountain_car_dqn/mountain_car_dqn_1.json'
    f8 = './results/policy_gradient/reinforce.json'

    window = 10
    metrics = ['reward', 'avg_reward', 'data_required', 'energy_required', 'latency_required', 'payment_required',
               'energy', 'latency', 'payment', 'data_1', 'data_2', 'actions', 'states']
    # for i in range(len(metrics)):
    #     plot_learning(f, window, metrics[i])
    ID = 4
    evaluated_metric = 'energy'
    file = './results/result_{}.json'.format(ID)
    plot_learning(file, window, evaluated_metric)
