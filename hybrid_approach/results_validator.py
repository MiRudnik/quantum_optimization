from collections import defaultdict

from qubo_matrices_helpers import create_avec, check_chosen_and_slack_qubits, get_cost


def validate_sample(ones_list, qubits_number, C, paths, tasks_number, machines_number, A, deadline):
    real_qubits_number = tasks_number * machines_number

    A_vec = create_avec(A, paths, real_qubits_number)
    # print("A_VEC: {}".format(A_vec))

    # paths_qubits = create_paths_qubits(paths)

    total_qubits_len = qubits_number
    # COMPUTE REAL PROBLEM ENERGY (NOT QUBO ENERGY)
    chosen_qubits = [int(q[1:]) for q in ones_list if int(q[1:]) < real_qubits_number]
    slack_qubits = [int(q[1:]) for q in ones_list if int(q[1:]) >= real_qubits_number]
    chosen_qubits_vector = [0 for _ in range(real_qubits_number)]
    for cq in chosen_qubits:
        chosen_qubits_vector[cq] = 1

    slack_mark, times = check_chosen_and_slack_qubits(chosen_qubits_vector, slack_qubits, total_qubits_len,
                                                      real_qubits_number, paths, deadline, A_vec, tasks_number,
                                                      machines_number)

    cost = get_cost(C, real_qubits_number, chosen_qubits_vector)

    # print("PATHS QUBITS {}\n".format(paths_qubits))
    # print("ENERGY: {}".format(cost))
    # print("TIMES: {}".format(time))

    tasks_number_mark = ""
    if len(list(filter(lambda x: int(x[1:]) < real_qubits_number, ones_list))) != tasks_number:
        tasks_number_mark = "WRONG TASKS NUMBER"
    return slack_mark + tasks_number_mark + str(list(filter(lambda x: int(x[1:]) < 200, ones_list))), cost, times
