import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite

from hybrid_approach.results_validator import validate_sample

import qubo_matrices_helpers

from utils.jobshop_helpers import ones_from_sample

check_gaps = True

# size8: P=8, S=10
# size10: P=14, S=25
# size15: P=11, S=10
# size18: P=6, S=40

filename = "test_results.txt"
P = 11
S = 10

A, b, C, paths, tasks_number, machines_number, deadline = qubo_matrices_helpers.get_8_qubits_data(S)
real_qubits_number = tasks_number * machines_number
# SOLUTION
D = np.diag(2 * A.transpose().dot(b))
QUBO = P * (A.transpose().dot(A) + D) + C
qubits_number = len(QUBO[0])
linear, quadratic = qubo_matrices_helpers.prepare_qubo_dicts_dwave(QUBO)
Q = dict(linear)
Q.update(quadratic)

maxval = max(list(Q.values()))
minval = min(list(Q.values()))

print(maxval)
print(minval)

# results_file.close()
cs_large = abs(maxval) if abs(maxval) > abs(minval) else abs(minval)
cs = cs_large/4.0
results_file = open(filename, "a+")

sampling_result = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=2000, return_embedding=True)
# print(sampling_result.info)
embedding = sampling_result.info['embedding_context']['embedding']
chain_strength = sampling_result.info['embedding_context']['chain_strength']
params_string = "P={} S={} chain_strength={}\n".format(P, S, chain_strength)
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
    info, cost, time = validate_sample(ones_from_sample(s.sample), qubits_number, C, paths, tasks_number,
                                       machines_number, A, deadline)
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
