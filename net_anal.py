import pandas as pd
import networkx as nx

# read dataset
df = pd.read_csv("athletes_edges_NA.csv")

# create graph object from df
G = nx.from_pandas_edgelist(df, source="node_1", target="node_2")

# computes  degree & eigenvector centrality for each node in the graph
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
for node in G.nodes():
    print(f"Node {node}: ")
    print(f"\tDegree Centrality: {degree_centrality[node]}")
    print(f"\tEigenvector Centrality: {eigenvector_centrality[node]}\n")

bridges = list(nx.bridges(G))
print(f"Bridges in Network: {bridges}\n")

# betweenness_centrality = nx.betweenness_centrality(G)
# edge_centrality = nx.edge_betweenness_centrality(G)