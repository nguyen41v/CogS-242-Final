import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import math

def graphFig1():
    increment = 0.1
    incrementByInverseOf = 10
    max_diff = 100
    difficulty = []
    for i in range(max_diff*incrementByInverseOf):
        difficulty.append(i * increment)
    before_delta = 30
    error_rate = []
    learning_rate = []

    after_delta = 15
    error_rate_f = []
    learning_rate_f = []

    for diff in difficulty:
        gaussian_curve = norm.cdf([-math.inf, 0], diff, before_delta)
        error_rate.append(gaussian_curve[1] - gaussian_curve[0])
        pdf_curve = norm.pdf([-math.inf, 0], diff, before_delta)
        learning_rate.append((pdf_curve[1] - pdf_curve[0]) * diff * before_delta)

        gaussian_curve_f = norm.cdf([-math.inf, 0], diff, after_delta)
        error_rate_f.append(gaussian_curve_f[1] - gaussian_curve_f[0])
        pdf_curve_f = norm.pdf([-math.inf, 0], diff, after_delta)
        learning_rate_f.append((pdf_curve_f[1] - pdf_curve_f[0]) * diff * after_delta)

    min_h = -100
    max_h = 125
    h = []
    for i in range((max_h - min_h) * incrementByInverseOf):
        h.append(min_h + increment * i)

    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(h, norm.pdf(h, 16, before_delta), 'b-', lw=2, alpha=1, label='Before learning')
    ax1.plot(h, norm.pdf(h, 16, after_delta), 'r-', lw=2, alpha=1, label='After learning')
    ax1.set_title('Fig. 1A of paper; distribution of decision variable h')
    ax1.set_xlabel('Decision variable, h')
    ax1.set_ylabel('Probability of h')
    ax1.set_xlim([min_h, max_h])
    ax1.set_ylim(bottom=0)

    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(difficulty, error_rate,'b-', lw=2, alpha=1, label='Before learning')
    ax2.plot(difficulty, error_rate_f, 'r-', lw=2, alpha=1, label='After learning')
    ax2.set_title('Fig. 1B of paper; error rate as a function of difficulty')
    ax2.set_xlabel('Difficulty')
    ax2.set_ylabel('Error rate, ER')
    ax2.set_xlim([0, max_diff])
    ax2.set_ylim(bottom=0)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(difficulty, learning_rate, 'b-', lw=2, alpha=1, label='Before learning')
    ax3.plot(difficulty, learning_rate_f, 'r-', lw=2, alpha=1, label='After learning')
    ax3.set_title('Fig. 1C of paper; learning rate as a function of difficulty')
    ax3.set_xlabel('Difficulty')
    ax3.set_ylabel('Learning rate, dER/dB')
    ax3.set_xlim([0, max_diff])
    ax3.set_ylim(bottom=0)

    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(error_rate, learning_rate, 'b-', lw=2, alpha=1, label='Before learning')
    ax4.plot(error_rate_f, learning_rate_f, 'r-', lw=2, alpha=1, label='After learning')
    ax4.set_title('Fig. 1D of paper; learning rate as a function of error rate')
    ax4.set_xlabel('Error rate, ER')
    ax4.set_ylabel('Learning rate, dER/dB')
    ax4.set_xlim([0, 0.5])
    ax4.set_ylim(bottom=0)

if __name__ == '__main__':
    graphFig1()
    plt.show()









