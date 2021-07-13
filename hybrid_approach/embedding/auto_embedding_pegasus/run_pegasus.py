import numpy as np
from dwave.embedding import pegasus
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

from hybrid_approach.results_validator import validate_sample

import qubo_matrices_helpers

from utils.jobshop_helpers import ones_from_sample

check_gaps = True

# 8 qubits: S=30, P=0.2
# 8 qubits: P=12,S=30,fact=0.0005,CS=4.0,AT=20
# 8q P8 S20 div 12000.0
# 8q P8 S10 no div CS1200
# 10 S=25,P=14,fact=0.000075,CS=0.5
# 10 S=25,P=14,fact=1.0,CS=6650.0
# 15 S=10, P=11, fact=1.0, CS=2800.0
# 18 S=25 P=6 fact=1 CS=14000

for S in [25]:
    A, b, C, paths, tasks_number, machines_number, deadline = qubo_matrices_helpers.get_10_qubits_data(S)
    real_qubits_number = tasks_number * machines_number
    for P in [14]:  # itertools.chain(range(3,13,3), range(17,35,5), range(45,76,15)):
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
        results_file = open("added_params_size_10_1.txt", "a+")

        ### V1
        # additional params: auto_scale=True, annealing_time=8, postprocess="optimization" - no such parameter
        sampling_result = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=2000, return_embedding=True, auto_scale=True, annealing_time=8)
        ### V2
        # sampler = DWaveSampler()
        # embedding = pegasus.find_clique_embedding(46, 6)  # Disconnected chain error
        # embedding = pegasus.find_clique_embedding(46, target_graph=sampler.to_networkx_graph())  # over 5mins without finishing
        # sampling_result = FixedEmbeddingComposite(sampler, embedding=embedding)\
        #     .sample_qubo(Q, num_reads=2000, return_embedding=True)
        # print(sampling_result.info)
        embedding = sampling_result.info['embedding_context']['embedding']
        chain_strength = sampling_result.info['embedding_context']['chain_strength']
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
