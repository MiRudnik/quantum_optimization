import dwave
from dwave.system import DWaveSampler
from dwave_networkx import chimera_graph

import qubo_matrices_helpers

import numpy as np

from hybrid_approach.new_embeddings import get_d2000_q6_embedding
from utils.jobshop_helpers import ones_from_sample

check_gaps = True

P = 6
S = 40
chain_strength = 6650
A, b, C, paths, tasks_number, machines_number, deadline = qubo_matrices_helpers.get_smaller_18_qubits_data(S)

# SOLUTION
D = np.diag(2 * A.transpose().dot(b))
QUBO = P * (A.transpose().dot(A) + D) + C

qubits_number = len(QUBO[0])
# linear, quadratic = qubo_matrices_helpers.prepare_qubo_dicts(QUBO)
linear, quadratic = qubo_matrices_helpers.prepare_qubo_dicts_dwave(QUBO)
Q = dict(linear)
Q.update(quadratic)

results_file = open("size_18_3.txt", "a+")
embedding = get_d2000_q6_embedding(qubits_number)
reverse_embedding = qubo_matrices_helpers.get_reverse_embedding(embedding)

emb_len = float(len(embedding['x0']))
tQ = dwave.embedding.embed_qubo(Q, embedding, chimera_graph(16), chain_strength=chain_strength)

# additional params: auto_scale=True, annealing_time=8, postprocess="optimization"
# https://docs.dwavesys.com/docs/latest/c_solver_1.html#postprocess
sampling_result = DWaveSampler(solver='DW_2000Q_6').sample_qubo(tQ, num_reads=2000,
                                                                # auto_scale=True, annealing_time=8,
                                                                # postprocess="optimization"
                                                                )

max_chain_len = max(len(chain) for chain in embedding.values())
params_string = "P={} S={} max_chain_len={} chain_strength={}\n".format(P, S, max_chain_len, chain_strength)
print(params_string)
results_file.write(params_string)
print(str(embedding))
results_file.write(str(embedding) + "\n")
list_sampling_result = list(sampling_result.data())
sum = 0
if check_gaps:
    gaps = 50.0
    min_num = list_sampling_result[0].energy
    max_num = list_sampling_result[len(list_sampling_result) - 1].energy
    diff = max_num - min_num
    gap = diff/int(gaps)
for s in list_sampling_result:
    sum += s.num_occurrences
    info, cost, time = qubo_matrices_helpers.check_sample(ones_from_sample(s.sample), embedding,
                                                          reverse_embedding, qubits_number, C, paths,
                                                          tasks_number, machines_number, A, deadline)
    deadline_mark = ""
    if max(time) > deadline:
        deadline_mark = "DEADLINE "
    res_str = "{}{}, COST: {}, TIME: {}, DEADLINE: {}, ENERGY: {}, OCCURENCES: {}/{}".format(deadline_mark,
                                                                                             info, cost, time,
                                                                                             deadline,
                                                                                             s.energy,
                                                                                             s.num_occurrences,
                                                                                             sum)
    print(res_str)
    results_file.write("{}\n".format(res_str))
if check_gaps:
    gaps_sizes = []
    samples_index = 0
    tmp_reference_energy = min_num
    for i in range(int(gaps)):
        counter = 0
        while samples_index != len(list_sampling_result) \
                and list_sampling_result[samples_index].energy < tmp_reference_energy + gap:
            counter += 1
            samples_index += 1
        gaps_sizes.append(counter)
        tmp_reference_energy += gap
    print(gaps_sizes)
    results_file.write("{}\n".format(gaps_sizes))
results_file.close()
