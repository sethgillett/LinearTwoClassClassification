import math
import random
from time import perf_counter

from autograd import grad
from autograd import numpy as np
from scipy import optimize as opt

from utilities import check_relabeling, no_misclassifications, check_result, plot_accuracy, plot_results

# CONSTANTS

# constant for logistic function in least squares cost function
t = 3
# size of training data set
training_data_set_size = 100
# size of test data set
test_data_set_size = 100
# maximum number of gradient descent iterations before failure (if force_perfect_training_classification is False)
max_grad_descent_iterations = 100
# number of times test data will be generated and tested against the algorithms
test_data_iterations = 1000
# whether to show plots of data and best fit lines
show_plots = True
# force gradient descent to achieve 100% correct classification in training data before proceeding
# WARNING: may cause program to hang
force_perfect_training_classification = False


# Least squares loss function
def least_squares_loss(w, data):
    total = 0
    for (x, y) in data:
        total += (logistic(t * np.dot(x, w)) - y) ** 2
    return total / len(data)


# Cross entropy loss function
def cross_entropy_loss(w, data):
    total = 0
    for (x, y) in data:
        total += y * np.log(logistic(np.dot(x, w))) + \
                 (1 - y) * np.log(1 - logistic(np.dot(x, w)))
    return -total / len(data)


# Soft-max loss function
def soft_max_loss(w, data):
    total = 0
    for (x, y) in data:
        total += np.log(1 + np.e ** (-y * np.dot(x, w)))
    return total / len(data)


# Logistic step function
def logistic(x):
    return 1 / (1 + np.e ** (-x))


# Generate training / test data
def gen_data(labels=(-1, 1), data_set_size=100, source_vector=None):
    w_length = 2
    if source_vector is None:
        # bias weight
        source_vector = [random.uniform(-50, 50)]
        # remaining weights
        for _ in range(w_length):
            source_vector.append(random.uniform(-100, 100))
        source_vector = np.array(source_vector)
    data = []
    for _ in range(data_set_size):
        # bias
        x = [1.0]
        # remaining x values
        for _ in range(w_length):
            x.append(random.uniform(-10, 10))
        x = np.array(x)
        y_val = np.dot(x, source_vector)
        assert y_val != 0
        if y_val < 0:
            y = labels[0]
        else:
            y = labels[1]
        data.append([x, y])
    return source_vector, data


def perceptron_learning(data):
    # w starts off as the zero vector
    w = np.zeros(len(data[0][0]))
    # misclassified is set to false at the beginning of each iteration
    # if any data is misclassified it will be reset to True
    misclassified = True
    misclassifications = 0
    iterations = 0

    start = perf_counter()

    while misclassified:
        misclassified = False
        for (x, y) in data:
            # Checks whether the data is correctly classified
            classification = math.copysign(y, np.dot(x, w))
            if classification != y:
                misclassified = True
                misclassifications += 1
                # if the output is not correct, misclassified is set to true and
                # w is adjusted accordingly
                w = w + y * x
        iterations += 1

    stop = perf_counter()

    print("Perceptron learning took {:.2f} seconds with {} iterations".format(stop-start, iterations))
    return w


def linprog_classifier(data):
    c = np.zeros(len(data[0][0]))
    A_ub = []
    for (x, y) in data:
        A_ub.append(-y * x)
    b_ub = []
    for _ in data:
        b_ub.append(-1)

    start = perf_counter()

    result = opt.linprog(c, A_ub, b_ub, bounds=(None, None))

    stop = perf_counter()

    print("LP classifier took {:.2f} seconds".format(stop-start))

    return result


def gradient_descent(loss_function, data, relabel=True):
    # Changes the labels on the data to (0, 1)
    if relabel:
        zero_one_data = [(x, (y+1)/2) for (x, y) in data]
        assert check_relabeling(data, zero_one_data, (-1, 1), (0, 1))
        # Creates lambda function for use in autograd
        f = lambda w: loss_function(w, zero_one_data)
    else:
        f = lambda w: loss_function(w, data)
    dfdx = grad(f)
    # Starts w at 0
    w = np.zeros(len(data[0][0]))
    count = 0

    start = perf_counter()

    alpha = 0.1
    while (not no_misclassifications(data, w)) and \
            (count < max_grad_descent_iterations or force_perfect_training_classification):
        slope = dfdx(w)
        w += - alpha * slope
        count += 1
        # Decrease alpha if solution is not found
        if count % 100 == 0:
            alpha = alpha / 2.0

    stop = perf_counter()
    print("Gradient descent took {:.2f} seconds, {} iterations, and {} "
          "misclassifications in training data".format(stop-start, count, check_result(data, w)))

    return w


def classify():
    # Perceptron Learning misclassifications
    pl_misclassifications = 0
    # Linear Program Classification misclassifications
    lc_misclassifications = 0
    # Gradient Descent with least squares loss function misclassifications
    gd_ls_misclassifications = 0
    # Gradient Descent with soft max loss function misclassifications
    gd_sm_misclassifications = 0
    # Gradient Descent with cross-entropy loss function misclassifications
    gd_ce_misclassifications = 0
    # Generate training data and source vector
    source_vector, training_data = gen_data(labels=(-1, 1), data_set_size=training_data_set_size)
    # Linprog classifier
    lc_res = linprog_classifier(training_data).x
    # Perceptron learning classifier
    pl_res = perceptron_learning(training_data)
    # Gradient descent with least squares loss function
    gd_ls_res = gradient_descent(least_squares_loss, training_data, relabel=True)
    # Gradient descent with soft max loss function
    gd_sm_res = gradient_descent(soft_max_loss, training_data, relabel=False)
    # Gradient descent with cross-entropy loss function
    gd_ce_res = gradient_descent(cross_entropy_loss, training_data, relabel=True)
    # Show plot at end
    labels = []
    labels.append("Source Vector")
    labels.append("Linear Classification")
    labels.append("Perceptron Learning")
    labels.append("Gradient Descent w/ Least Squares")
    labels.append("Gradient Descent w/ Soft Max")
    labels.append("Gradient Descent w/ Cross-Entropy")
    for _ in range(test_data_iterations):
        # Generate test data with the same source vector
        _, test_data = gen_data(labels=(-1, 1), data_set_size=test_data_set_size, source_vector=source_vector)
        lc_misclassifications += check_result(test_data, lc_res)
        pl_misclassifications += check_result(test_data, pl_res)
        gd_ls_misclassifications += check_result(test_data, gd_ls_res)
        gd_sm_misclassifications += check_result(test_data, gd_sm_res)
        gd_ce_misclassifications += check_result(test_data, gd_ce_res)
    results = [lc_misclassifications, pl_misclassifications, gd_ls_misclassifications,
               gd_sm_misclassifications, gd_ce_misclassifications]
    total_data_size = test_data_set_size * test_data_iterations
    percentage_results = [(total_data_size - result)/total_data_size for result in results]
    print(f"\n{test_data_iterations} test data sets were generated with {test_data_set_size} data points each")
    for label, result in zip(labels[1:], percentage_results):
        print("In total, {} classified data correctly {:.1%} of the time".format(label, result))
    if show_plots:
        plot_results(training_data, [source_vector, lc_res, pl_res, gd_ls_res, gd_sm_res, gd_ce_res],
                     labels,
                     "Vectors Found in Training Data")
        plot_accuracy(labels[1:], percentage_results)


classify()
