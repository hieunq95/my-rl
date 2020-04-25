from __future__ import division
import gym
import json
from TD_learning.sarsa import SARSA

TEST_ID = 1
discrete_space = 20
nb_games = 20000

logger = {
    'scores': [],
    'epsilon': [],
    'episode': [],
}

file_name = './results/mountain_car_sarsa_{}.json'.format(TEST_ID)
window = 100


def to_discrete_state(state, p0, v0, window_p, window_v):
    p = state[0]
    v = state[1]
    p_offset = int((p - p0 ) / window_p)
    v_offset = int((v - v0) / window_v)
    return p_offset*discrete_space + v_offset


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    action_space = env.action_space.n
    state_space = discrete_space**2

    agent = SARSA(nb_states=state_space, nb_actions=action_space, alpha=0.1, gamma=0.99,
                  epsilon1=0.1, epsilon_min1=0.1, epsilon_decay1=1e-5,
                  epsilon2=0.1, epsilon_min2=0.1, epsilon_decay2=1e-3)
    position0 = env.observation_space.low[0]
    velocity0 = env.observation_space.low[1]
    window_p = (env.observation_space.high[0] - env.observation_space.low[0]) / discrete_space
    window_v = (env.observation_space.high[1] - env.observation_space.low[1]) / discrete_space

    print(action_space, state_space, env.observation_space.sample(), env.action_space, env.observation_space,
          env.observation_space.low, env.observation_space.high)

    print('******************* Mountain_car_sarsa test: {} begins ********************** \n'.format(TEST_ID))
    for i in range(nb_games):
        done = False
        state = env.reset()
        action = agent.choose_action(to_discrete_state(state, position0, velocity0, window_p, window_v), next_max=False)
        score = 0
        steps = 0
        while not done:
            s = to_discrete_state(state, position0, velocity0, window_p, window_v)
            next_state, reward, done, _ = env.step(action)
            s_ = to_discrete_state(next_state, position0, velocity0, window_p, window_v)
            next_action = agent.choose_action(s_, next_max=True)
            agent.update_q_table(reward, action, s, next_action, s_, done)
            state = next_state
            action = next_action

            score += reward
            steps += 1
        agent.epsilon_update(next_max=True)

        logger['episode'].append(i + 1)
        logger['epsilon'].append(agent.epsilon2)
        logger['scores'].append(score)

        if i >= window and i % window == 0:
            with open(file_name, 'w') as outfile:
                json.dump(logger, outfile)
            print('Episode: {}, epsilon1: {}, epsilon2: {}, steps: {}, scores: {}'
                  .format(i+1, agent.epsilon1, agent.epsilon2, steps, score))
    env.close()
    print('******************* Mountain_car_q_learning test: {} ends ********************** \n'.format(TEST_ID))
