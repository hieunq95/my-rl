import gym
import json
from n_steps_bootstrapping.n_steps_sarsa import NstepsSarsa

TEST_ID = 1

discrete_space = 20
nb_games = 20000
file_name = './results/mountain_car_nsteps_sarsa_{}.json'.format(TEST_ID)
window = 100

logger = {
    'scores': [],
    'epsilon': [],
    'episode': [],
}


def to_discrete_state(state, p0, v0, window_p, window_v):
    p = state[0]
    v = state[1]
    p_offset = int((p - p0 ) / window_p)
    v_offset = int((v - v0) / window_v)
    return p_offset*discrete_space + v_offset


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    action_space = env.action_space.n
    state_space = discrete_space ** 2
    p0 = env.observation_space.low[0]
    v0 = env.observation_space.low[1]
    w_p = (env.observation_space.high[0] - env.observation_space.low[0]) / discrete_space
    w_v = (env.observation_space.high[1] - env.observation_space.low[1]) / discrete_space

    agent = NstepsSarsa(nb_states=state_space, nb_actions=action_space,
                        epsilon=1.0, epsilon_min=0.05, epsilon_decay=1e-3,
                        alpha=0.05, gamma=0.99, n=20, mem_size=1001)
    print(agent.q_table.shape)
    print('******************* Mountain_car_nsteps_sarsa test: {} begins ********************** \n'.format(TEST_ID))
    for i in range(nb_games):
        s0 = to_discrete_state(env.reset(), p0, v0, w_p, w_v)
        agent.store_in_memory(s0, 0, 0)
        # choose action a0
        a0 = agent.choose_action(s0)
        agent.store_in_memory(a0, 0, 1)
        T = 1000
        tau = 0

        scores = 0
        steps = 0
        for t in range(T):
            if t < T:
                at = agent.memory[1, t]
                st_, rt_, done, _ = env.step(at)
                st_ = to_discrete_state(st_, p0, v0, w_p, w_v)
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
                agent.update_epsilon()
                # print(agent.memory)
                break

        logger['episode'].append(i + 1)
        logger['epsilon'].append(agent.epsilon)
        logger['scores'].append(scores)

        if i % window == 0:
            with open(file_name, 'w') as outfile:
                json.dump(logger, outfile)
            print('Episode: {}, epsilon: {}, steps: {}, scores: {}'.format(i, agent.epsilon, steps, scores))

    print('******************* Mountain_car_nsteps_sarsa test: {} ends ********************** \n'.format(TEST_ID))


