from qubo_matrices_helpers import find_complete_graph_embedding

embedding_15 = find_complete_graph_embedding(33)
embedding_18 = find_complete_graph_embedding(46)


print("33 bits:")
print("max_chain_len:", max(len(chain) for chain in embedding_15.values()))
print(embedding_15)

print("46 bits:")
print("max_chain_len:", max(len(chain) for chain in embedding_18.values()))
print(embedding_18)
