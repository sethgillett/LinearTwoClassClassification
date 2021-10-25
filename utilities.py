import math
import string

from autograd import numpy as np
from matplotlib import pyplot as plt


# Debug function to check whether data has been properly relabeled
def check_relabeling(old_data, new_data, old_labels, new_labels):
    for i in range(len(old_data)):
        old_label = old_data[i][1]
        idx = old_labels.index(old_label)
        new_label = new_labels[idx]
        if new_data[i][1] != new_label:
            return False
    return True


# Returns the number of misclassifications with a given vector and data set
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


def plot_accuracy(labels, results):
    y_pos = np.arange(len(labels))
    heights = [height*100 for height in results]
    labels = [get_acronym(label) for label in labels]
    fig, ax = plt.subplots()

    hbars = ax.barh(y_pos, heights, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Accuracy of Linear Classification Algorithms')

    # Label with specially formatted floats
    ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(left=min(heights)-10, right=101)

    plt.show()


def get_acronym(label):
    return "".join([c for c in label if c in string.ascii_uppercase])
