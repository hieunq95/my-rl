import gym
import json
from dqn.dqn_agent import Agent

TEST_ID = 2
file_name = './results/mountain_car_dqn/mountain_car_dqn_{}.json'.format(TEST_ID)

logger = {
    'episode': [],
    'scores': [],
    'epsilon': [],
    'steps': [],
}


if __name__ == '__main__':
    nb_games = 600
    env = gym.make('MountainCar-v0')
    input_dims = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    scores = []
    load_model = True
    h5_file = './results/mountain_car_dqn/mountain_car_model.h5'

    agent = Agent(alpha=0.001, gamma=0.99, n_actions=nb_actions, epsilon=0.9, epsilon_end=0.1, epsilon_dec=0.08,
                  batch_size=64, input_dims=input_dims, mem_size=500000, replace=1000)
    print(env.observation_space.shape[0], env.action_space, env.observation_space,
          env.observation_space.low, env.observation_space.high)

    print('******************* Mountain_car_dqn test: {} begins ********************** \n'.format(TEST_ID))

    if load_model:
        agent.load_model(h5_file)

    for i in range(nb_games):
        state = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if not load_model:
                agent.learn()
            steps += 1
            score += reward
        scores.append(score)
        agent.update_epsilon()
        logger['scores'].append(score)
        logger['epsilon'].append(agent.epsilon)
        logger['episode'].append(i + 1)
        logger['steps'].append(steps)

        with open(file_name, 'w') as outfile:
            json.dump(logger, outfile)

        print('Episode: {}, epsilon: {}, steps: {}, scores: {}'.format(i+1, agent.epsilon, steps, score))
    agent.save_model(h5_file)
    print('******************* Mountain_car_dqn test: {} ends ********************** \n'.format(TEST_ID))