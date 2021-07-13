import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

# fig = plt.figure(figsize=(15, 15))
# params = {"node_size": 10, "width": 1.0, "edge_color": 'gray', "node_color": 'r'}

G = dnx.chimera_graph(16, 16, 4)
print("Num nodes:", len(G.nodes))
print("Num edges:", len(G.edges))
# dnx.draw_chimera(G, **params)

# plt.show()
# plt.savefig('foo.png')

# ______________________________________________________________________________

# fig = plt.figure(figsize=(30, 30))
# params = {"node_size": 7, "width": 1.0, "edge_color": 'gray', "node_color": 'r'}

# P = dnx.pegasus_graph(16)
# print("Num nodes:", len(P.nodes))
# print("Num edges:", len(P.edges))
# dnx.draw_pegasus(P, **params)

# plt.show()
# plt.savefig('foo.png')
