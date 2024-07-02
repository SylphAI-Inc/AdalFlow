from pyvis.network import Network
import networkx as nx

# Create a simple NetworkX graph
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4)])

# Create a Pyvis Network
net = Network(notebook=False, width="100%", height="100%", directed=True)

# Add nodes and edges from NetworkX to Pyvis
net.from_nx(G)

# Save the network to an HTML file
net.show("test_network.html")
