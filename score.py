import argparse
import numpy as np
import scipy.linalg as la
import networkx as nx
from math import floor
from statistics import mean
import itertools

np.set_printoptions(linewidth=200)

debug = True
const = 0.5


def log(text):
    if debug:
        print("[DBG] " + text)


def remove_node_numbers(matrix):
    return matrix[np.ix_(range(1, matrix.shape[0]), range(1, matrix.shape[0]))]


def add_node_numbers_to_laplacian(laplacian, preset_numbers=None):
    if preset_numbers is None:
        half_size = laplacian.shape[0]//2
        node_numbers = [i for i in range(1, half_size+1)]
        node_numbers = node_numbers + node_numbers
    else:
        node_numbers = list(preset_numbers)
    laplacian = np.insert(laplacian, 0, node_numbers, axis=0)
    node_numbers = [0] + node_numbers
    laplacian = np.insert(laplacian, 0, node_numbers, axis=1)
    return laplacian


def assert_on_bad_zero_value(eigenvalues):
    zero_value_positions = np.where(abs(eigenvalues.real) < 0.0000001)[0]
    if len(zero_value_positions) == 0:
        print("Incorrect laplacian matrix!")
        exit(1)
    log("Zero eigenvalue at position: " + str(zero_value_positions[0]))


def get_node_numbers(laplacian):
    return laplacian[0:][:1][0][1:]


def module_to_indices(laplacian, module):
    node_numbers = get_node_numbers(laplacian)
    node_numbers = node_numbers[0:len(node_numbers)//2]
    indices = list()
    for node in module:
        indices.append(list(node_numbers).index(node))
    return indices


def indices_to_module(laplacian, indices):
    node_numbers = get_node_numbers(laplacian)
    node_numbers = node_numbers[0:len(node_numbers)//2]
    return [node_numbers[index] for index in indices]


def get_q1_from_laplacian(laplacian):
    size = laplacian.shape[0] - 1
    half_size = size//2
    q1 = laplacian[np.ix_(range(1, half_size+1), range(half_size+1, size+1))]
    return q1


def calc_density(laplacian, indices):
    indices = list(indices)
    q1 = get_q1_from_laplacian(laplacian)
    q1_module = np.abs(q1[np.ix_(indices, indices)])
    q1_module_sum = q1_module.sum()
    density = q1_module_sum/pow(len(indices), 2)
    return density


def delete_module_from_laplacian(laplacian, module):
    half_size = (laplacian.shape[0]-1)//2
    indices = np.array(module_to_indices(laplacian, module))+1
    indices = list(indices) + list(np.array(indices)+half_size)
    step_one = np.delete(laplacian, indices, 0)
    step_two = np.delete(step_one,  indices, 1)
    node_numbers = get_node_numbers(step_two)
    step_three = -nx.laplacian_matrix(nx.from_numpy_matrix(remove_node_numbers(step_two))).toarray()
    step_four = add_node_numbers_to_laplacian(step_three, preset_numbers=node_numbers)
    return step_four


def get_modules(laplacian):
    laplacian = add_node_numbers_to_laplacian(laplacian)
    log("The Laplacian matrix:\n" + str(laplacian))

    discovered_modules = []
    size = laplacian.shape[0]-1  # Because of the added node numbers
    previous_size = 0
    last = False

    while 1 < size != previous_size:
        log("The Laplacian matrix:\n" + str(laplacian))

        # Check if the current matrix isn't already a single module first
        initial_nodes = get_node_numbers(laplacian)
        initial_nodes = initial_nodes[0:len(initial_nodes)//2]
        initial_density = calc_density(laplacian, module_to_indices(laplacian, initial_nodes))
        log("Initial density = " + str(initial_density))
        if initial_density < const:

            # Calculate eigenvectors and eigenvalues
            eigenvalues, eigenvectors = la.eig(remove_node_numbers(laplacian))
            log("Eigenvalues: " + str(eigenvalues.real.round(4)))

            # Drop if we have no zero eigenvalue - means bad Laplacian matrix was provided
            assert_on_bad_zero_value(eigenvalues)

            # Find the Fiedler value and from it the Fiedler vector by index
            fiedler_val = np.sort(eigenvalues.real)[1]
            fiedler_pos = np.where(eigenvalues.real == fiedler_val)[0][0]
            log("Smallest eigenvalue: " + str(fiedler_val.round(4)) + ", located at position: " + str(fiedler_pos))
            fiedler_eigenvector = np.transpose(eigenvectors)[fiedler_pos]
            log("Fiedler eigenvector: \n" + str(fiedler_eigenvector.round(4)))
            fiedler_eigenvector = np.array([val if abs(val) > 0.003 else 0 for val in fiedler_eigenvector])
            log("Fiedler eigenvector after filtering: \n" + str(fiedler_eigenvector.round(4)))

            # Get the split rule for two modules from the Fiedler vector
            half_fiedler_length = int(len(fiedler_eigenvector) / 2)
            modules = [np.sign(fiedler_eigenvector[x]) == np.sign(fiedler_eigenvector[x + half_fiedler_length]) and
                       np.sign(fiedler_eigenvector[x]) < 0 for x in range(half_fiedler_length)]
            modules_inverse = list(np.invert(modules))
            log("Modules 1 = " + str(modules))
            log("Modules 2 = " + str(modules_inverse))

            # Get the two modules themselves from the split rule
            module_choice_1 = indices_to_module(laplacian, np.where(modules)[0])
            module_choice_2 = indices_to_module(laplacian, np.where(modules_inverse)[0])
            module_choice = list()
            if len(module_choice_1) > 0:
                module_choice.append(module_choice_1)
            if len(module_choice_2) > 0:
                module_choice.append(module_choice_2)
            log("Module choices are = " + str(list(module_choice)))

            # Calculate the densities of the two modules
            densities = [calc_density(laplacian, module_to_indices(laplacian, choice)) for choice in module_choice]
            log("Module densities are " + str(densities))

            # Calculate the size of the two modules
            sizes = [len(choice) for choice in list(module_choice)]
            log("Sizes = " + str(sizes))

            # Find the module with the best density
            best_density_module_index = densities.index(max(densities))

            # Find the module with the best size (we prefer bigger modules)
            best_size_module_index = sizes.index(max(sizes))

            # Decide on the best module, size first, but density has to be good
            module = module_choice[best_size_module_index]
            density = densities[best_size_module_index]
            if density < const:
                module = module_choice[best_density_module_index]
                density = densities[best_density_module_index]
            log("And the winner is: " + str(module) + " with density of " + str(density))

            # Delete the good module from the Laplacian for further calculations
            if len(module_choice) > 1:
                laplacian = delete_module_from_laplacian(laplacian, module)
            else:
                last = True
        else:
            module = list(initial_nodes)

        # We have a good module, save it
        if not last:
            discovered_modules.append(module)
        else:
            # We're left with unit size modules, add them one by one
            for m in module:
                discovered_modules.append([m])
                log("Added module " + str(m))
            laplacian = np.array([])

        # Update the Laplacian size variables
        previous_size = size
        size = laplacian.shape[0] - 1
        log("size = {}, previous_size = {}".format(size, previous_size))

    print("Discovered modules = " + str(discovered_modules))
    return discovered_modules


def calc_sum_from_matrix(matrix, indices):
    module = np.abs(matrix[np.ix_(indices, indices)])
    module_sum = module.sum()
    return module_sum


def calc_sum_from_partitions(matrix, indices1, indices2):
    part1 = np.abs(matrix[np.ix_(indices1, indices2)])
    part2 = np.abs(matrix[np.ix_(indices2, indices1)])
    part_sum = part1.sum() + part2.sum()
    return part_sum


def calc_score(laplacian, modules):
    # Get the first quarter of the laplacian matrix
    laplacian = add_node_numbers_to_laplacian(laplacian)
    matrix = get_q1_from_laplacian(laplacian)
    log("Matrix = \n" + str(matrix))

    # Get all module pair combinations
    num_of_modules = len(modules)
    log("num_of_modules = " + str(num_of_modules))
    modules_index = range(1, num_of_modules+1)
    index_combs = list(itertools.combinations(modules_index, 2))
    module_combs = list(itertools.combinations(modules, 2))
    log("index_combs = " + str(index_combs))
    log("module_combs = " + str(module_combs))

    # Get internal_edges for each combination
    internal_edges = list()
    for comb in module_combs:
        sum_for_module1 = calc_sum_from_matrix(matrix, module_to_indices(laplacian, comb[0]))
        sum_for_module2 = calc_sum_from_matrix(matrix, module_to_indices(laplacian, comb[1]))
        internal_edges.append(sum_for_module1 + sum_for_module2)
    log("internal_edges = " + str(internal_edges))

    # Get external edges for each combination
    external_edges = list()
    for comb in module_combs:
        sum_for_module_comb = calc_sum_from_partitions(matrix,
                                                       module_to_indices(laplacian, comb[0]),
                                                       module_to_indices(laplacian, comb[1]))
        external_edges.append(sum_for_module_comb)
    log("external_edges = " + str(external_edges))

    # Get total_cells for each combination
    total_cells = list()
    for comb in module_combs:
        sum_for_module_comb = pow(len(comb[0]) + len(comb[1]), 2)
        total_cells.append(sum_for_module_comb)
    log("total_cells = " + str(total_cells))

    # Calculate max_edges for each combination
    max_edges = list()
    for cells in total_cells:
        max_edges.append(floor(cells * const))
    log("max_edges = " + str(max_edges))

    # Calculate max_extra_edges for each combination
    max_extra_edges = list()
    for i in range(len(module_combs)):
        max_extra_edges.append(max_edges[i] - internal_edges[i])
    log("max_extra_edges = " + str(max_extra_edges))

    # Calculate the score for each combination
    score = list()
    for i in range(len(module_combs)):
        if max_extra_edges[i] != 0:
            score.append(external_edges[i]/max_extra_edges[i])
        else:
            score.append(1)
    log("score = " + str(score))

    average_score = mean(score)
    log("average = " + str(average_score))

    for i in range(len(module_combs)):
        print("For modules {} the score is {}%".format(index_combs[i], int(score[i]*100)))
    print("The average score is {}%".format(int(average_score*100)))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Matrix file name")
    args = parser.parse_args()

    laplacian = np.loadtxt(args.file, dtype=int)

    modules = get_modules(laplacian)

    calc_score(laplacian, modules)


if __name__ == '__main__':
    run()
