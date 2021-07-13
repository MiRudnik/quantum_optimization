import dimod
import numpy as np

import qubo_matrices_helpers

from hybrid.decomposers import ComponentDecomposer
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

print("_______________________________")

decomposer = ComponentDecomposer(key=len)
state1 = decomposer.next(state0).result()
print(len(state1.subproblem.variables))
