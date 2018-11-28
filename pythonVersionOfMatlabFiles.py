from __future__ import print_function


import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import math
import csv


def graph_fig1():
    """Generates 4 plots similar to the 4 plots in Fig. 1 of the paper"""
    steps = 0.1
    increment_by_inverse_of = 10
    max_diff = 100
    difficulty = []

    noise = 30
    error_rate = []
    learning_rate = []

    noise_f = 15
    error_rate_f = []
    learning_rate_f = []

    # getting values for plots with given deltas
    for i in range(max_diff*increment_by_inverse_of):
        diff = i * steps
        difficulty.append(diff)

        gaussian_curve = norm.cdf([-math.inf, 0], diff, noise)
        error_rate.append(gaussian_curve[1] - gaussian_curve[0])
        pdf_curve = norm.pdf([-math.inf, 0], diff, noise)
        learning_rate.append((pdf_curve[1] - pdf_curve[0]) * diff * noise)

        gaussian_curve_f = norm.cdf([-math.inf, 0], diff, noise_f)
        error_rate_f.append(gaussian_curve_f[1] - gaussian_curve_f[0])
        pdf_curve_f = norm.pdf([-math.inf, 0], diff, noise_f)
        learning_rate_f.append((pdf_curve_f[1] - pdf_curve_f[0]) * diff * noise_f)

    min_h = -100
    max_h = 125
    h = []
    for i in range((max_h - min_h) * increment_by_inverse_of):
        h.append(min_h + steps * i)

    # plot Fig. 1A
    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(h, norm.pdf(h, 16, noise), 'b-', lw=1, alpha=1, label='Before learning')
    ax1.plot(h, norm.pdf(h, 16, noise_f), 'r-', lw=1, alpha=1, label='After learning')
    ax1.set_title('Fig. 1A of paper; distribution of decision variable h')
    ax1.set_xlabel('Decision variable, h')
    ax1.set_ylabel('Probability of h')
    ax1.set_xlim([min_h, max_h])
    ax1.set_ylim(bottom=0)

    # plot Fig. 1B
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(difficulty, error_rate,'b-', lw=1, alpha=1, label='Before learning')
    ax2.plot(difficulty, error_rate_f, 'r-', lw=1, alpha=1, label='After learning')
    ax2.set_title('Fig. 1B of paper; error rate as a function of difficulty')
    ax2.set_xlabel('Difficulty,  ∆')
    ax2.set_ylabel('Error rate, ER')
    ax2.set_xlim([0, max_diff])
    ax2.set_ylim(bottom=0)

    # plot Fig. 1C
    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(difficulty, learning_rate, 'b-', lw=1, alpha=1, label='Before learning')
    ax3.plot(difficulty, learning_rate_f, 'r-', lw=1, alpha=1, label='After learning')
    ax3.set_title('Fig. 1C of paper; learning rate as a function of difficulty')
    ax3.set_xlabel('Difficulty,  ∆')
    ax3.set_ylabel('Learning rate, ∂ER/∂β')
    ax3.set_xlim([0, max_diff])
    ax3.set_ylim(bottom=0)

    # plot Fig. 1D
    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(error_rate, learning_rate, 'b-', lw=1, alpha=1, label='Before learning')
    ax4.plot(error_rate_f, learning_rate_f, 'r-', lw=1, alpha=1, label='After learning')
    ax4.set_title('Fig. 1D of paper; learning rate as a function of error rate')
    ax4.set_xlabel('Error rate, ER')
    ax4.set_ylabel('Learning rate, ∂ER/∂β')
    ax4.set_xlim([0, 0.5])
    ax4.set_ylim(bottom=0)


def my_basic_perceptron(num_of_trials=10, num_of_sims=1000, num_of_tests=1000):
    """Creates a Teacher Perceptron to generate true labels

    https://en.wikipedia.org/wiki/Perceptron
    The Teacher Perceptron weight vector is randomly generated.
    Untrained Perceptron weight vectors are also randomly generated
        ("to mimic a modest degree of initial training").
    This function basically just shows that the untrained Perceptrons get trained.
    It'll print out the accuracy of the trained and untrained models after all training has been complete.

    :param int num_of_trials: number of Perceptrons to make and train
    :param int num_of_sims: number of training simulations to do on an untrained Perceptron
    :param int num_of_tests: number of tests to do on trained and untrained Perceptron weight vectors
    """

    # generate true weight vector of Teacher Perceptron
    e = unit_vector(np.random.normal(0, 1, 100))

    all_acc_before = np.array([])
    all_acc_after = np.array([])
    for o in range(num_of_trials):
        w = np.random.normal(0, 1, 100)

        # make a copy for comparing error rates
        original_w = np.copy(w)
        for i in range(num_of_sims):

            # generate stimulus
            x = np.random.normal(0, 1, 100)

            # calculate decision variable
            h = np.dot(w, x)

            # map onto label
            y = 0
            if h > 0:
                y = 1

            # calculate true decision variable and label
            real = np.dot(e, x)
            t = 0
            if real > 0:
                t = 1

            # update weights if predicted label is incorrect
            w = w + (t - y) * x

        num_correct_after = 0
        num_correct_before = 0
        for k in range(num_of_tests):
            # generate stimulus
            x = np.random.normal(0, 1, 100)

            # calculate true decision variable and label
            real = np.dot(e, x)
            t = 0
            if real > 0:
                t = 1

            # calculate decision variable and label before learning
            h_before = np.dot(original_w, x)
            y_before = 0
            if h_before > 0:
                y_before = 1
            if y_before == t:
                num_correct_before += 1

            # calculate decision variable and label after learning
            h_after = np.dot(w, x)
            y_after = 0
            if h_after > 0:
                y_after = 1
            if y_after != t:
                num_correct_after += 1

        all_acc_before = np.append(all_acc_before, num_correct_before/num_of_tests)
        all_acc_after = np.append(all_acc_after, num_correct_after/num_of_tests)

    print('accuracy before:', all_acc_before,
          '\naccuracy after:', all_acc_after)

    print('accuracy before: ',np.average(all_acc_before) / num_of_trials,
          '\naccuracy after: ', np.average(all_acc_after) / num_of_trials)


def unit_vector(vector):
    """
    From: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    From: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def run_sims_for_error_rate(e, error_rate, w, my_lambda, num_of_sims, num_of_tests=1000):
    """
    Trains a Perceptron model at a fixed error rate and
    writes the F1 score of the model with each training stimulus

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param float error_rate: training error rate [0, 1]
    :param numpy.ndarray w: initial weight vector of untrained Perceptron
    :param float my_lambda: arbitrary value to keep the difficulty constant
    :param int num_of_sims: number of training simulations to do on w
    :param int num_of_tests: number of simulations to do for F1 score
    :returns: a numpy.ndarray of the trained Perceptron weight vector
    """
    row = [round(error_rate, 2)]
    row.append(test_F1(e, w, num_of_sims))
    accuracy = 0
    for i in range(num_of_sims):
        my_theta = angle_between(w, e)
        x = np.random.normal(0, 1, 100)

        real = np.dot(e, x)
        t = 0
        if real > 0:
            t = 1

        my_delta = abs(norm.ppf(error_rate) * my_lambda * math.tan(my_theta))
        signal = np.linalg.norm(w) * (2 * t - 1) * my_delta * math.cos(my_theta)
        noise = np.linalg.norm(w) * np.dot(
            ((w / np.linalg.norm(w)) - e * math.cos(my_theta)), x)
        h = signal + noise

        y = 0
        if h > 0:
            y = 1
        if y == t:
            accuracy += 1

        w = w + (t - y) * x
        row.append(test_F1(e, w, num_of_tests))
    row.insert(0, 1 - accuracy/num_of_sims)
    print(abs(error_rate - (1 - accuracy/num_of_sims)))
    return row

# did not check for division by 0; should not matter since
# data should be large enough so that there is no division
# by 0
def test_accuracy(e, w, num_of_tests):
    """Returns the accuracy of Perceptron w

    https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    accuracy = (true positives + true negatives) /
               (true positives + true negatives + false positives + false negatives)

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param numpy.ndarray w: weight vector of Perceptron
    :param int num_of_tests: number of tests to simulate
    :returns: a float of accuracy, [0, 1]
    """
    correct = 0
    for k in range(num_of_tests):
        x = np.random.normal(0, 1, 100)

        real = np.dot(e, x)
        t = 0
        if real > 0:
            t = 1

        h = np.dot(w, x)
        y = 0
        if h > 0:
            y = 1
        if y == t:
            correct += 1
    return correct / num_of_tests


def test_precision(e, w, num_of_tests):
    """Returns the precision of Perceptron w

    https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    precision = (true positives) /
                (true positives + false positives)

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param numpy.ndarray w: weight vector of Perceptron
    :param int num_of_tests: number of tests to simulate
    :returns: a float of precision, [0, 1]
    """
    true_positives = 0
    predicted_positives = 0
    for k in range(num_of_tests):
        x = np.random.normal(0, 1, 100)

        real = np.dot(e, x)
        t = 0
        if real > 0:
            t = 1

        h = np.dot(w, x)
        y = 0
        if h > 0:
            y = 1

        if y == 1:
            predicted_positives += 1
            if y == t:
                true_positives += 1

    return true_positives/predicted_positives


def test_absolute_precision(e, w, num_of_tests):
    """Returns the absolute precision of Perceptron w

    Absolute precision is measured as the inverse of the standard
    deviation of the noise.

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param numpy.ndarray w: weight vector of Perceptron
    :param int num_of_tests: number of tests to simulate
    :returns: a float of precision, [0, 1]
    """
    min = 100
    max = 0
    noises = []
    for k in range(num_of_tests):
        x = np.random.normal(0, 1, 100)

        real = np.dot(e, x)
        t = 0
        if real > 0:
            t = 1
        my_theta = angle_between(unit_vector(w), unit_vector(e))
        noise = np.linalg.norm(w) * np.dot(
            ((w / np.linalg.norm(w)) - e * math.cos(my_theta)), x)
        noises.append(noise)
        h = np.dot(w, x)
        y = 0
        if h > 0:
            y = 1
        h = abs(h)
        if h > max:
            max = h
        if h < min:
            min = h
    return 1/np.std(np.array(noises))


def test_recall(e, w, num_of_tests):
    """Returns the recall of Perceptron w

    https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    recall = (true positives) /
             (true positives + false negatives)

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param numpy.ndarray w: weight vector of Perceptron
    :param int num_of_tests: number of tests to simulate
    :returns: a float of recall, [0, 1]
    """
    true_positives = 0
    actual_positives = 0
    for k in range(num_of_tests):
        x = np.random.normal(0, 1, 100)
        real = np.dot(e, x)
        t = 0
        if real > 0:
            t = 1

        h = np.dot(w, x)
        y = 0
        if h > 0:
            y = 1

        if t == 1:
            actual_positives += 1
            if y == t:
                true_positives += 1
    return true_positives / actual_positives


def test_F1(e, w, num_of_tests):
    """Returns the F1 score of Perceptron w

    https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    F1 score = 2 * (precision * recall) / (precision + recall)

    :param numpy.ndarray e: normalized weight vector of Teacher Perceptron
    :param numpy.ndarray w: weight vector of Perceptron
    :param int num_of_tests: number of tests to simulate
    :returns: a float of F1 score, [0, 1]
    """
    precision = test_precision(e, w, num_of_tests)
    recall = test_recall(e, w, num_of_tests)
    return 2 * (precision*recall)/(precision + recall)


def fixed_training_error_perceptron(filename='data.csv', min_error_rate=0.01, max_error_rate=0.5,
                                    steps=0.01, num_of_trials=100, num_of_sims=1000, num_of_tests=1000):
    """Generates training data for a Perceptron trained at a fixed error rate.

    The results are written in a csv format to filename.
    The untrained Perceptron is randomly generated every trial.

    :param string filename: name/path of file to save test data to
    :param float min_error_rate: minimum error rate to check (0, 0.5]
    :param float max_error_rate: maximum error rate to check (0, 0.5]
    :param float steps: the size of steps between different error rates
    :param int num_of_trials: number of untrained Perceptrons to train per error rate
    :param int num_of_sims: number of training simulations to do per untrained Perceptron
    :param int num_of_tests: number of tests to do to see the untrained Perceptron's performance
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for o in range(num_of_trials):
            current_error_rate = min_error_rate
            while current_error_rate <= max_error_rate + steps/2:
                # generate weight vector of Teacher Perceptron
                e = unit_vector(np.random.normal(0, 1, 100))
                w = np.random.normal(0, 1, 100)
                # keep tan(theta) and cos(theta) positive; think this was the constraint in the paper (<1.6 radians)
                while angle_between(unit_vector(w), e) > math.pi/2:
                    w = np.random.normal(0, 1, 100)
                my_lambda = 1
                writer.writerow(run_sims_for_error_rate(e, current_error_rate, w, my_lambda, num_of_sims, num_of_tests=num_of_tests))
                print('Trial %d of %d for error rate of %f has been completed'%(o + 1, num_of_trials, current_error_rate))
                current_error_rate += steps


def fixed_training_error_perceptron2(filename='data.csv', min_error_rate=0.01, max_error_rate=0.5,
                                    steps=0.01, num_of_trials=100, num_of_sims=1000, num_of_tests=1000):
    """Generates training data for a Perceptron trained at a fixed error rate.

    The results are written in a csv format to filename.
    The untrained Perceptron is randomly generated every trial.
    Same as other function, except it goes from max to min error rate

    :param string filename: name/path of file to save test data to
    :param float min_error_rate: minimum error rate to check (0, 0.5]
    :param float max_error_rate: maximum error rate to check (0, 0.5]
    :param float steps: the size of steps between different error rates
    :param int num_of_trials: number of untrained Perceptrons to train per error rate
    :param int num_of_sims: number of training simulations to do per untrained Perceptron
    :param int num_of_tests: number of tests to do to see the untrained Perceptron's performance
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for o in range(num_of_trials):
            current_error_rate = max_error_rate
            while current_error_rate >= min_error_rate - steps/2:
                # generate weight vector of Teacher Perceptron
                e = unit_vector(np.random.normal(0, 1, 100))
                w = np.random.normal(0, 1, 100)
                # keep tan(theta) and cos(theta) positive; think this was the constraint in the paper (<1.6 radians)
                while angle_between(unit_vector(w), e) > math.pi/2:
                    w = np.random.normal(0, 1, 100)
                my_lambda = 1
                writer.writerow(run_sims_for_error_rate(e, current_error_rate, w, my_lambda, num_of_sims, num_of_tests=num_of_tests))
                print('Trial %d of %d for error rate of %f has been completed'%(o + 1, num_of_trials, current_error_rate))
                current_error_rate -= steps


def calculate_stats(filein='F1analysis.csv', fileout_averages='averages.csv', fileout_stds='stds.csv', data_size=100):
    """Writes the average and std of csv data into another csv file

    Input file should be organized in that the rows are trials and the
    columns are error rates, with each cell being the F1 score (or any
    other measure). The columns should be sorted so that all occurrences
    (trials) of error rate x are all next to each other. Each error rate
    should have the same number of occurrences

    :param string fileout_stds: name of csv file to write standard deviations to
    :param string filein: name of csv file with data
    :param fileout_averages: name of csv file to write averages to
    :param data_size: number of trials/occurances of every error rate
    """
    with open(fileout_averages, 'a', newline='') as csvfile1:
        avg_writer = csv.writer(csvfile1)
        with open(fileout_stds, 'a', newline='') as csvfile2:
            std_writer = csv.writer(csvfile2)
            with open(filein, 'r', newline='') as csvfile3:
                reader = csv.reader(csvfile3)
                for row in reader:
                    average = []
                    std = []
                    for i in range(len(row)//data_size):
                        error_rate_data = np.array(row[i * data_size : (i + 1) * data_size]).astype(np.float)
                        average.append(np.average(error_rate_data))
                        std.append(np.std(error_rate_data))
                        print(np.average(error_rate_data), np.std(error_rate_data))
                    avg_writer.writerow(average)
                    std_writer.writerow(std)




if __name__ == '__main__':
    graph_fig1()
    plt.show()
    fixed_training_error_perceptron2(filename='test1414.csv', steps=0.01, num_of_trials=10000, min_error_rate=0.01, max_error_rate=0.5, num_of_sims=1000)
