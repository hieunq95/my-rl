import numpy as np
import json
from environment.block_fl import BlockFLEnv

logger = {
    'actions': []
}

file_name = './results/random_agent.json'

actions = [i for i in range(16384)]

for i in range(16384 * 100):
    a = np.random.choice(actions)
    # print(a)
    logger['actions'].append(int(a))

with open(file_name, 'w') as outfile:
    json.dump(logger, outfile)