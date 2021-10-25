import math
import random

from matplotlib import pyplot as plt
from scipy import optimize as opt
from autograd import grad
from autograd import numpy as np
from time import perf_counter


# CONSTANTS

# logistic function constant
t = 3
# size of training data set
data_set_size = 100
# maximum number of gradient descent iterations before failure
max_grad_descent_iterations = 100
# number of times test data will be generated and tested against the algorithms
test_data_iterations = 100
# whether to show plots of data and best fit lines
show_plots = False


# Debug function to check whether data has been properly relabeled
def check_relabeling(old_data, new_data, old_labels, new_labels):
    for i in range(len(old_data)):
        old_label = old_data[i][1]
        idx = old_labels.index(old_label)
        new_label = new_labels[idx]
        if new_data[i][1] != new_label:
            return False
    return True


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
    return 1 / (1 + math.e ** (-x))


# Generate training / test data
def gen_data(labels=(-1, 1), source_vector=None):
    # w_length = random.randint(2, 10)
    w_length = 2
    if source_vector is None:
        # bias weight
        # w = [random.uniform(-0.5, 0.5)]
        source_vector = [random.uniform(-50, 50)]
        # remaining weights
        for _ in range(w_length):
            # w.append(random.uniform(-5, 5))
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


# def logistic_regression(loss_function, data):
#     w = np.zeros(len(data[0][0]))
#     loss_grad = grad(loss_function)
#     for _ in range(grad_descent_iterations):
#         grad_neg = loss_grad(w)
#         print(grad_neg)


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

    print(f"Perceptron learning took {stop - start} seconds with {iterations} iterations")
    return w


def check_result(data, w):
    misclassifications = 0
    for (x, y) in data:
        classification = math.copysign(y, np.dot(x, w))
        if classification != y:
            misclassifications += 1
    return misclassifications


# Algorithm performs significantly faster when checking with this function
def no_misclassifications(data, w):
    for (x, y) in data:
        classification = math.copysign(y, np.dot(x, w))
        if classification != y:
            return False
    return True


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

    print(f"LP classifier took {stop - start} seconds")

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

    while not no_misclassifications(data, w) and count < max_grad_descent_iterations:
        # Adjusts the alpha to be smaller as the program progresses
        alpha = 1 / (count + 1)
        slope = dfdx(w)
        w += - alpha * slope
        count += 1

    stop = perf_counter()
    print(f"Gradient descent took {stop - start} seconds, {count} iterations, and {check_result(data, w)} "
          f"misclassifications in training data")

    return w


def scatter_color(x_values, y_values):
    plt.scatter(x_values, y_values, c=np.arange(0, 255, 255 / (len(x_values))))
    plt.savefig('scatter-with-color.png', bbox_inches='tight')
    plt.show()


def plot_results(data, vectors, labels, title):
    x_values = []
    y_values = []
    colors = []
    for (x, y) in data:
        x_values.append(x[1])
        y_values.append(x[2])
        if y > 0:
            colors.append('C1')
        else:
            colors.append('C2')
    plt.scatter(x_values, y_values, c=colors)
    for v in range(len(vectors)):
        vector = vectors[v]
        label = labels[v]
        vector_x_values = [min(x_values), max(x_values)]
        vector_y_values = []
        for x in vector_x_values:
            # a + bx + cy = 0
            # y = - (bx + a) / c
            vector_y_values.append(-(vector[0] + x * vector[1]) / vector[2])
            # source_vector_y_values.append(x)
        plt.plot(vector_x_values, vector_y_values, label=label, color=f'C{v}')
    leg = plt.legend(loc='best', ncol=2, mode="expand")
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    plt.show()


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
    source_vector, training_data = gen_data(labels=(-1, 1))
    # Linprog classifier
    print("Starting linear programming classifier")
    lc_res = linprog_classifier(training_data).x
    # Perceptron learning classifier
    print("Starting perceptron learning classifier")
    pl_res = perceptron_learning(training_data)
    # Gradient descent with least squares loss function
    print("Starting gradient descent with least squares loss function")
    gd_ls_res = gradient_descent(least_squares_loss, training_data, relabel=True)
    # Gradient descent with soft max loss function
    print("Starting gradient descent with soft max loss function")
    gd_sm_res = gradient_descent(soft_max_loss, training_data, relabel=False)
    # Gradient descent with cross-entropy loss function
    print("Starting gradient descent with cross entropy loss function")
    gd_ce_res = gradient_descent(cross_entropy_loss, training_data, relabel=True)
    # Show plot at end
    titles = [
                 "Source Vector",
                 "Linear Classification",
                 "Perceptron Learning",
                 "Gradient Descent w/ Least Squares",
                 "Gradient Descent w/ Soft Max",
                 "Gradient Descent w/ Cross-Entropy"
             ],
    if show_plots:
        plot_results(training_data, [source_vector, lc_res, pl_res, gd_ls_res, gd_sm_res, gd_ce_res],
                     titles,
                     "Vectors Found")
    for _ in range(test_data_iterations):
        # Generate test data with the same source vector
        _, test_data = gen_data(labels=(-1, 1), source_vector=source_vector)
        lc_misclassifications += check_result(test_data, lc_res)
        pl_misclassifications += check_result(test_data, pl_res)
        gd_ls_misclassifications += check_result(test_data, gd_ls_res)
        gd_sm_misclassifications += check_result(test_data, gd_sm_res)
        gd_ce_misclassifications += check_result(test_data, gd_ce_res)
    print()


classify()
