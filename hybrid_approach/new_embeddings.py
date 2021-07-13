from collections import defaultdict

import qubo_matrices_helpers


def get_d2000_q6_embedding(qubits_number):
    """
    Transform old embedding: start from top-left -> start from top-right
    D-Wave 2000Q is missing 7 nodes: [43, 46, 524, 548, 1723, 1735, 1804]
    """
    embedding = qubo_matrices_helpers.find_complete_graph_embedding(qubits_number)
    new_embedding = {}
    for key, qubits_set in embedding.items():
        new_qubits_set = {(int(x // 128) + 1) * 128 - ((x % 128) // 8) * 8 - (8 - (x % 8)) for x in qubits_set}
        new_embedding[key] = new_qubits_set
    return new_embedding


def get_reverse_embedding(embedding):
    result = defaultdict()
    for key in embedding.keys():
        for value in embedding[key]:
            result[value] = key
    return result
