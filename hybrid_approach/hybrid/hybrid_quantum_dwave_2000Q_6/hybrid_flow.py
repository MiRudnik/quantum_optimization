import logging

import dimod
import hybrid
import numpy as np
from dwave.system import DWaveSampler

import qubo_matrices_helpers

# logging
RESULT_FILENAME = "hybrid_flow_results_size15_subproblem16_pfs_2.log"
hybrid_logger = logging.getLogger('hybrid')
hybrid_logger.setLevel(logging.DEBUG)
logger = logging.getLogger('hybrid_like_flow')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(RESULT_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
hybrid_logger.addHandler(fh)
logger.addHandler(fh)

# Construct a problem
from hybrid_approach.results_validator import validate_sample
from utils.jobshop_helpers import ones_from_sample

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

# Define the workflow
iteration = hybrid.EnergyImpactDecomposer(size=16, traversal='pfs') | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=500, qpu_sampler=DWaveSampler(solver='DW_2000Q_6')) | hybrid.SplatComposer()
workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=4)

# Solve the problem
init_state = hybrid.State.from_problem(bqm)
final_state = workflow.run(init_state).result()

# Print results
print("Solution: sample={.samples.first}".format(final_state))
logger.info("Solution: sample={.samples.first}".format(final_state))

info, cost, time = validate_sample(ones_from_sample(final_state.samples.first.sample), qubits_number, C, paths,
                                   tasks_number, machines_number, A, deadline)
deadline_mark = ""
if max(time) > deadline:
    deadline_mark = "DEADLINE "
res_str = "{}{}, COST: {}, TIME: {}, DEADLINE: {}, ENERGY: {}".format(deadline_mark, info, cost, time,
                                                                      deadline, final_state.samples.first.energy)
logger.info(res_str)
print(res_str)
