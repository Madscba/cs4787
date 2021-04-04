#!/usr/bin/env python3
from scipy.special import softmax
from matplotlib import pyplot
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
from tqdm import tqdm
import copy
import time
matplotlib.use('agg')


mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory,
                                 return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:, perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:, perm])
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    x = Xs[:, ii]  # choose the datapoints
    y = Ys[:, ii]  # choose the corresponding targets
    softmax_input = np.matmul(W, x)
    un_reg_grad = np.matmul((softmax(softmax_input, axis=0)-y), x.T)
    l2_reg_grad = gamma * W
    return (un_reg_grad) / len(ii) + l2_reg_grad  # Average over the ii samples


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    incorrect_count = 0
    Xs = Xs.T
    Ys = Ys.T
    for x, y in zip(Xs, Ys):
        softmax_input = np.matmul(W, x)
        pred = np.argmax(softmax(softmax_input))
        incorrect_count += 1 if y[pred] == 0 else 0
    return incorrect_count / np.shape(Xs)[0]


# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    (d, n) = Xs.shape
    softmax_input = np.matmul(W, Xs)
    Ys_hat = softmax(softmax_input, axis=0)
    total_loss = - np.sum(Ys * np.log(Ys_hat)) / n
    return total_loss


def multinomial_logreg_total_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d, n) = Xs.shape
    softmax_input = np.matmul(W, Xs)
    Ys_hat = softmax(softmax_input, axis=0)
    total_grad = np.matmul((Ys_hat-Ys), Xs.T)
    total_grad = total_grad / n
    return total_grad + gamma * W

# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs


def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 1
    res = []
    error = []
    W_i = np.array(W0, copy=True)
    for i in tqdm(range(num_epochs)):
        if (i % monitor_period == 0):
            res.append(copy.deepcopy(W_i))
        diff = -alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W_i)
        W_i += diff
    res.append(W_i)
    return res

# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs


def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    res = []
    error = []
    W_i = np.array(W0, copy=True)
    V_i = np.array(W0, copy=True)
    for i in tqdm(range(num_epochs)):
        if (i % monitor_period == 0):
            res.append(copy.deepcopy(W_i))
        diff = -alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W_i)
        V_prev = V_i
        V_i = W_i + diff
        W_i = V_i + beta * (V_i - V_prev)
    res.append(W_i)
    return res

# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches


def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 2
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            ii = list(range(i * B, i * B + B))
            if ((t*(n/B)+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
            W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(copy.deepcopy(W))
    return res

# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches


def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    V = 0
    res = []
    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            ii = list(range(i * B, i * B + B))
            if ((t*(n/B)+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
            V = beta * V - alpha * \
                multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
    res.append(copy.deepcopy(W))
    return res

# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches


def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    return 0

# Evaluates the training error, test error, and training loss


def eval_tr_te_loss(variations, gamma, Xs_tr, Ys_tr, Xs_te, Ys_te):
    n = len(variations[0])
    m = len(variations)
    errors_tr = np.zeros((m, n))
    errors_te = np.zeros((m, n))
    loss_tr = np.zeros((m, n))
    for i in tqdm(range(m)):
        errors_tr[i, :] = [multinomial_logreg_error(
            Xs_tr, Ys_tr, weights) for weights in variations[i]]
        print("Finished evaluating training error for variation ", i)
        errors_te[i, :] = [multinomial_logreg_error(
            Xs_te, Ys_te, weights) for weights in variations[i]]
        print("Finished evaluating testing error for variation ", i)
        loss_tr[i, :] = [multinomial_logreg_loss(
            Xs_tr, Ys_tr, gamma, weights) for weights in variations[i]]
        print("Finished evaluating training loss for variation ", i)
    return errors_tr, errors_te, loss_tr


def plot_tr_te_loss(plot_data, num_epochs, legend, part):
    for i in range(3):
        matplotlib.pyplot.close()
        matplotlib.pyplot.annotate
        matplotlib.pyplot.figure(figsize=(9, 7))
        matplotlib.pyplot.title("{}".format(
            ["Training Error", "Test Error", "Training Loss"][i]))
        matplotlib.pyplot.xlabel("Number of epochs")
        matplotlib.pyplot.ylabel("{}".format(["Error", "Error", "Loss"][i]))
        X = np.linspace(0, num_epochs+1, num_epochs+1)
        data = plot_data[i]
        for j in range(data.shape[0]):
            matplotlib.pyplot.plot(X, plot_data[i][j, :])
        matplotlib.pyplot.legend([legend[0] + ", final: {:.3f}".format(plot_data[i][0, -1]),
                                  legend[1] +
                                  ", final: {:.3f}".format(
                                      plot_data[i][1, -1]),
                                  legend[2] + ", final: {:.3f}".format(plot_data[i][2, -1])])
        matplotlib.pyplot.savefig(
            "Part"+str(part)+"_{}.png".format(["TrainingError", "TestError", "TrainingLoss"][i]))
        # matplotlib.pyplot.show()


def part_1(Xs_tr, Ys_tr, Xs_te, Ys_te):
    # 6. Gradient Descent, Nesterov 0.9, Nesterov 0.99
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    W0 = np.random.normal(0, 1, size=(c, d))
    gamma = 0.0001
    alpha = 1.0
    beta1 = 0.9
    beta2 = 0.99
    num_epochs = 100
    monitor_period = 1
    gd_w = gradient_descent(Xs_tr, Ys_tr, gamma, W0,
                            alpha, num_epochs, monitor_period)
    gdn_9_w = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha,
                          beta1, num_epochs, monitor_period)
    gdn_99_w = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha,
                           beta2, num_epochs, monitor_period)
    # 7. Evaluating training, test, loss for each model
    p1_variations = [gd_w, gdn_9_w, gdn_99_w]
    p1_errors_tr, p1_errors_te, p1_loss_tr = eval_tr_te_loss(
        p1_variations, gamma, Xs_tr, Ys_tr, Xs_te, Ys_te)
    # 8. Plot training, test, and loss
    plot_data = [p1_errors_tr, p1_errors_te, p1_loss_tr]
    legend = ["Gradient descent",
              "Nesterov's momentum with β = 0.9",
              "Nesterov's momentum with β = 0.99"]
    plot_tr_te_loss(plot_data, num_epochs, legend, 1)
    # 9. Run each algorithm 5 times (4 more) and average the time
    num_runs = 5
    sum_time1 = 0
    sum_time2 = 0
    for run in range(num_runs):
        t1 = time.time()
        gd_w = gradient_descent(Xs_tr, Ys_tr, gamma, W0,
                                alpha, num_epochs, monitor_period)
        sum_time1 += time.time() - t1
        t2 = time.time()
        gdn_9_w = gd_nesterov(Xs_tr, Ys_tr, gamma, W0,
                              alpha, beta1, num_epochs, monitor_period)
        sum_time2 += time.time() - t2
    avg_time1 = sum_time1/num_runs
    avg_time2 = sum_time2/num_runs
    print("Average times: ")
    print(avg_time1, avg_time2)
    # 10. Testing hyperparameters


def part_2(Xs_tr, Ys_tr, Xs_te, Ys_te):
    # 3. Stochastic Gradient Descent, Momentum 0.9, Momentum 0.99
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    W0 = np.random.normal(0, 1, size=(c, d))
    gamma = 0.0001
    alpha = 0.2
    B = 600
    beta1 = 0.9
    beta2 = 0.99
    num_epochs = 10
    monitor_period = 100
    sgd_mss_w = sgd_minibatch_sequential_scan(
        Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    sgd_mss_9_w = sgd_mss_with_momentum(
        Xs_tr, Ys_tr, gamma, W0, alpha, beta1, B, num_epochs, monitor_period)
    sgd_mss_99_w = sgd_mss_with_momentum(
        Xs_tr, Ys_tr, gamma, W0, alpha, beta2, B, num_epochs, monitor_period)
    # 4. Evaluating training, test, loss for each model
    p2_variations = [sgd_mss_w, sgd_mss_9_w, sgd_mss_99_w]
    p2_errors_tr, p2_errors_te, p2_loss_tr = eval_tr_te_loss(
        p2_variations, gamma, Xs_tr, Ys_tr, Xs_te, Ys_te)
    # 5. Plot training, test, and loss
    plot_data = [p2_errors_tr, p2_errors_te, p2_loss_tr]
    legend = ["Stochastic gradient descent",
              "Momentum with SGD, β = 0.9",
              "Momentum with SGD, β = 0.99"]
    plot_tr_te_loss(plot_data, num_epochs, legend, 2)
    # 6. Run each algorithm 5 times (4 more) and average the time
    num_runs = 5
    sum_time1 = 0
    sum_time2 = 0
    for run in range(num_runs):
        t1 = time.time()
        sgd_mss_w = sgd_minibatch_sequential_scan(
            Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
        sum_time1 += time.time() - t1
        t2 = time.time()
        sgd_mss_9_w = sgd_mss_with_momentum(
            Xs_tr, Ys_tr, gamma, W0, alpha, beta1, B, num_epochs, monitor_period)
        sum_time2 += time.time() - t2
    avg_time1 = sum_time1/num_runs
    avg_time2 = sum_time2/num_runs
    print("Average times: ")
    print(avg_time1, avg_time2)
    # 7. Testing hyperparameters


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    part_1(Xs_tr, Ys_tr, Xs_te, Ys_te)
    part_2(Xs_tr, Ys_tr, Xs_te, Ys_te)
