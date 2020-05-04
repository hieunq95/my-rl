import networkx as nx
import h5py

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_edge(1, 2)
G.add_edges_from([(1, 2), (1, 3)])

G.number_of_nodes()
G.number_of_edges()

file_name = '../experiments/results/mountain_car_dqn/mountain_car_model.h5'

with h5py.File(file_name, "r") as f:
    # List all groups
    print("Key: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    weights = a_group_key
    # Get the data
    data = list(f[a_group_key])
    print(weights)



