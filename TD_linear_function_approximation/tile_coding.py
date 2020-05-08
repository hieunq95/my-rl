"""
Tile coding for 2 dimensional continuous space, i.e., MountainCar-v0
Reference: https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_tilings(tilings):
    """Plot each tiling as a grid."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid('off')
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    return ax  # return Axis object to draw on later, if needed


def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins or tiles along each corresponding dimension.
        offsets : tuple
            Split points for each dimension should be offset by these values.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        """
    dim1 = np.linspace(low[0], high[0], bins[0]) + offsets[0]
    dim2 = np.linspace(low[1], high[1], bins[1]) + offsets[1]
    return [dim1, dim2]


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tiling_grid().

        Returns
        -------
        tilings : list
            A list of tilings (grids), each produced by create_tiling_grid().
        """
    tilings = []
    for i in tiling_specs:
        tilings.append(create_tiling_grid(low, high, bins=i[0], offsets=i[1]))
    return tilings


def discrete(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    intervalX = grid[0][1] - grid[0][0]
    intervalY = grid[1][1] - grid[1][0]
    xCoord = int((sample[0] - grid[0][0]) // intervalX)
    yCoord = int((sample[1] - grid[1][0]) // intervalY)
    if xCoord < 0 or yCoord < 0:
        raise Exception('Failed to create tilings, change offsets and try again')
    return [xCoord, yCoord]


def tile_encode(sample, action, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    action: discrete action
        A discrete action number on the agent that will be tiled together with the sampled space
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    encoded_sample = []
    vector_list = []
    vector_len = len(tilings[0][0]) - 1

    for tile in tilings:
        encoded_sample.append(discrete(sample, tile))
    # print('endocded_sample: ', encoded_sample)
    for e in encoded_sample:
        binVec = np.zeros(vector_len**3, dtype=int)
        binVec.put(e[0] + e[1]*vector_len + action*vector_len**2, 1)
        vector_list.append(binVec)

    if not flatten:
        return vector_list
    else:
        return np.array(vector_list).flatten()


"""
Usage example:
low = env.observation_space.low
high = env.observation_space.high
grid = create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0))
print(grid)

# Tiling specs: [(<bins>, <offsets>), ...]
delta_offset1 = 0.025
delta_offset2 = 1.944e-3
tiling_specs = [((10, 10), (0.0, 0.0)),
                ((10, 10), (-1*delta_offset1, -1*delta_offset2)),
                ((10, 10), (-2*delta_offset1, -2*delta_offset2)),
                ((10, 10), (-3*delta_offset1, -3*delta_offset2))]

tilings = create_tilings(low, high, tiling_specs)

state_sample = env.observation_space.sample()
discreted_sample = discrete(state_sample, grid)
ax = visualize_tilings(tilings)
plt.show(ax)

"""

