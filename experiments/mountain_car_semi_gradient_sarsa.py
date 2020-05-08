from __future__ import division
import numpy as np
import gym
import json
from TD_linear_function_approximation.semi_gradient_sarsa import SemiGradientSarsa
from TD_linear_function_approximation.tile_coding import create_tilings


logger = {
    'scores': [],
    'epsilon': [],
    'episode': [],
}

file_name = './results/mountain_car_semi_gradient_sarsa/result.json'


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    nb_episodes = 1000

    # Create 8 tilings and their corresponding offsets
    delta_offset1 = (1.8 / 18) / 8  # ((high - low) / (2*(bins - 1))) / nb_tilings
    delta_offset2 = (0.14 / 18) / 8
    # Tiling specs: [(<bins>, <offsets>), ...]
    tiling_specs = [((10, 10), (0.0, 0.0)),
                    ((10, 10), (-1 * delta_offset1, -1 * delta_offset2)),
                    ((10, 10), (-2 * delta_offset1, -2 * delta_offset2)),
                    ((10, 10), (-3 * delta_offset1, -3 * delta_offset2)),
                    ((10, 10), (-4 * delta_offset1, -4 * delta_offset2)),
                    ((10, 10), (-5 * delta_offset1, -5 * delta_offset2)),
                    ((10, 10), (-6 * delta_offset1, -6 * delta_offset2)),
                    ((10, 10), (-7 * delta_offset1, -7 * delta_offset2))
                    ]
    tilings = create_tilings([-1.2, -0.07], [0.6, 0.07], tiling_specs)

    agent = SemiGradientSarsa(nb_states=nb_states, nb_actions=nb_actions, w_dims=5832, alpha=0.2/8, gamma=0.99,
                              tilings=tilings, epsilon=0.1, epsilon_end=0.1, epsilon_decay=8e-3)
    feature_vector = np.zeros((agent.w_dims, nb_states, nb_actions), dtype=float)

    print(nb_states, nb_actions)
    print('******************* Mountain_car_semi_gradient_sarsa test: begins ********************** \n')

    for i in range(nb_episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False
        scores = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            w = agent.weights_update(reward, state, action, next_state, next_action, done)
            # print(w)
            state = next_state
            action = next_action

            scores += reward

        agent.update_epsilon()

        logger['episode'].append(i + 1)
        logger['epsilon'].append(agent.epsilon)
        logger['scores'].append(scores)

        if i >= 10 and i % 10 == 0:
            with open(file_name, 'w') as outfile:
                json.dump(logger, outfile)

        print('Episode: {}, Epsilon: {}, Scores: {}'.format(i, agent.epsilon, scores))

    print('******************* Mountain_car_semi_gradient_sarsa test: begins ********************** \n')