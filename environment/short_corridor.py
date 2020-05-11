"""
Implementation of short corridor environment. Details in example 13.1 - R. S. Sutton: "Reinforcement learning:
An introduction - 2nd edition"
"""
import gym


class ShortCorridor(gym.Env):
    def __init__(self):
        self.reset()

    def step(self, action):
        """
        Take an action and observe the transition
        :param action: 1 to go right, 0 to go left
        :return: tuple (reward, observation, done, info)
        """
        state = self.state
        if self.state == 0 or self.state == 2:
            if action == 1:
                state += 1
            else:
                state = max(0, self.state - 1)
        if self.state == 1:
            if action == 1:
                state += -1
            else:
                state += 1
        if self.state == 3:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.state = state
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
