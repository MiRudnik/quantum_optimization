import logging

import hybrid
import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler

from hybrid_approach.results_validator import validate_sample

import qubo_matrices_helpers

from utils.jobshop_helpers import ones_from_sample

# logging
RESULT_FILENAME = "tests_march/size_18/subproblem_30/test_results_30_subproblem3.log"
hybrid_logger = logging.getLogger('hybrid')
hybrid_logger.setLevel(logging.DEBUG)
logger = logging.getLogger('hybrid_test')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(RESULT_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
hybrid_logger.addHandler(fh)
logger.addHandler(fh)

# run parameters
run_params = {'max_subproblem_size': 30}

for S in [40]:
    A, b, C, paths, tasks_number, machines_number, deadline = qubo_matrices_helpers.get_18_qubits_data(S)
    real_qubits_number = tasks_number * machines_number
    for P in [6]:  # itertools.chain(range(3,13,3), range(17,35,5), range(45,76,15)):
        # SOLUTION
        D = np.diag(2 * A.transpose().dot(b))
        QUBO = P * (A.transpose().dot(A) + D) + C
        qubits_number = len(QUBO[0])
        linear, quadratic = qubo_matrices_helpers.prepare_qubo_dicts_dwave(QUBO)
        Q = dict(linear)
        Q.update(quadratic)

        params_string = "P={} S={} {}".format(P, S, run_params)

        logger.info(params_string)
        sampling_result = hybrid.KerberosSampler().sample_qubo(Q, **run_params)

        list_sampling_result = list(sampling_result.data())
        for s in list_sampling_result:
            info, cost, time = validate_sample(ones_from_sample(s.sample), qubits_number, C, paths, tasks_number,
                                               machines_number, A, deadline)

            deadline_mark = ""
            if max(time) > deadline:
                deadline_mark = "DEADLINE "
            res_str = "{}{}, COST: {}, TIME: {}, DEADLINE: {}, ENERGY: {}".format(deadline_mark, info, cost, time,
                                                                                  deadline, s.energy)
            logger.info(res_str)
