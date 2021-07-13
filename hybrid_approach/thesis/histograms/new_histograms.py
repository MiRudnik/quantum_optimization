import matplotlib.pyplot as plt
import numpy as np

problem_number = 1
computer_name = "Advantage"
res = [351, 290, 285, 159, 115, 150, 51, 67, 47, 49, 31, 15, 11, 31, 70, 79, 51, 43, 13, 14, 13, 13, 11, 3, 3, 4, 4, 2, 6, 3, 2, 4, 2, 0, 1, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

correct_number = 28
file_name = f'base_advantage_p{problem_number}.png'


def get_correct_and_wrong_list(source, correct_number):
    if not correct_number:
        return [0], source
    tmp = correct_number
    elements_number = 0
    while tmp > 0:
        tmp -= source[elements_number]
        elements_number += 1
    correct_solutions = source[:elements_number]
    if len(correct_solutions) > 0:
        correct_solutions[-1] += tmp
    wrong_solutions = source[(elements_number - 1):]
    return correct_solutions, wrong_solutions


correct, wrong = get_correct_and_wrong_list(res, correct_number)
# correct = [49, 91-49]   # problem4 postprocessing
correct = [float(x)/float(sum(res)) for x in correct]
wrong = [float(x)/float(sum(res)) for x in wrong]


def prepare_histogram(correct, wrong, ax, id):
    total_len = len(correct) + int(len(wrong)) - 1

    # ax.bar(np.arange(len(wrong)), wrong, color='red', label='błędne rozwiązania')  # problem4 postprocessing
    ax.bar(np.arange(len(correct) - 1, total_len), wrong, color='red', label='błędne rozwiązania')
    ax.bar(np.arange(len(correct)), correct, color='green', label='poprawne rozwiązania')
    ax.legend(loc='best')
    ax.set_title("Problem {} ({})".format(id, computer_name))
    ax.set_ylabel('Gęstość prawdopodobieństwa')
    ax.set_xlabel('Energia')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # ax.set_axis_off()
    # ax.patch.set_visible(False)


fig, ax1 = plt.subplots()
prepare_histogram(correct, wrong, ax1, problem_number)

fig.tight_layout(pad=0.75)
plt.savefig('imgs/' + file_name)
# plt.show()
