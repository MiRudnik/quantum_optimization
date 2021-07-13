import dimod
import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler
from hybrid import SplatComposer

import qubo_matrices_helpers

from hybrid.decomposers import EnergyImpactDecomposer
from hybrid.core import State
from hybrid.utils import random_sample

P = 11
S = 10
A, b, C, paths, tasks_number, machines_number, deadline = qubo_matrices_helpers.get_15_qubits_data(S)

real_qubits_number = tasks_number * machines_number
D = np.diag(2 * A.transpose().dot(b))
QUBO = P * (A.transpose().dot(A) + D) + C
qubits_number = len(QUBO[0])
linear, quadratic = qubo_matrices_helpers.prepare_qubo_dicts_dwave(QUBO)
Q = dict(linear)
Q.update(quadratic)

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
state0 = State.from_sample(random_sample(bqm), bqm)
print(len(state0.problem.variables))

print("__________________________________________")

decomposer = EnergyImpactDecomposer(size=10, rolling=True)
state1 = decomposer.next(state0).result()
print(state1.subproblem.variables)

state2 = decomposer.next(state1).result()
print(state2.subproblem.variables)

state3 = decomposer.next(state2).result()
print(state3.subproblem.variables)

state4 = decomposer.next(state3).result()
print(state4.subproblem.variables)

# print(state1.subproblem)
# print("__________________________________________")
#
# max_chain_len = 0
# composer = SplatComposer()
# new_state = state0
# for s in [state1, state2, state3, state4]:
#
#     res = EmbeddingComposite(DWaveSampler()).sample(s.subproblem, num_reads=500, return_embedding=True)
#     curr_max_len = max(len(chain) for chain in res.info['embedding_context']['embedding'].values())
#     if curr_max_len > max_chain_len:
#         max_chain_len = curr_max_len
#     best = res.first
#     print("BEST FOR", s.subproblem.variables)
#     print(best)
#     print("Before:", new_state)
#     new_state = new_state.updated(subsamples=best)
#     print("After:", new_state)
#
# composed = composer.run(new_state).result()
# print("Max chain len:", max_chain_len)
# print(composed)
