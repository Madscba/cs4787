import os
import numpy as np
import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('TkAgg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
from scipy.special import softmax
from tqdm import tqdm
import copy

def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    x = Xs[:, ii] #choose the datapoints
    y = Ys[:, ii] #choose the datapoints

    softmax_input = np.matmul(W,x)
    un_reg_grad = np.matmul((softmax(softmax_input)-y),x.T)
    l2_reg_grad = gamma * W
    return (un_reg_grad + l2_reg_grad) / len(ii) #Average over the ii samples
    # TODO students should implement this


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a fraction of incorrect labels (a number between 0 and 1)
def multinomial_logreg_error(Xs, Ys, W):
    incorrect_count = 0
    Xs = Xs.T
    Ys = Ys.T
    for x, y in zip(Xs, Ys):
        softmax_input = np.matmul(W, x)
        pred = np.argmax(softmax(softmax_input))

        incorrect_count += 1 if y[pred] == 0 else 0
    return incorrect_count / np.shape(Xs)[0]
    # TODO students should use their implementation from programming assignment 1


# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    _, n = Xs.shape
    num_epochs_corrected = num_epochs * n
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs_corrected)):
        datapoint_idx = random.sample(list(range(n)),1)
        W = W + alpha*multinomial_logreg_grad_i(Xs, Ys, datapoint_idx, gamma, W)
        if (t % monitor_period == 0):
            res.append(copy.deepcopy(W))
    return res
    # TODO students should implement this


# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    W = copy.deepcopy(W0)
    _, n = Xs.shape
    res = []
    for t in tqdm(range(num_epochs)):
        for i in range(n):
            W = W + alpha*multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W)
            if ((t*n+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
    return res
    # TODO students should implement this


# ALGORITHM 3: run stochastic gradient descent with minibatching
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
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    _, n = Xs.shape
    num_epochs_corrected = int(num_epochs * (n/B)) #10*60000/60 = 10000
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs_corrected)):
        datapoint_idx = random.sample(list(range(n)),B)
        W = W + alpha*multinomial_logreg_grad_i(Xs, Ys, datapoint_idx, gamma, W)
        if (t % monitor_period == 0):
            res.append(copy.deepcopy(W))
    return res
    # TODO students should implement this


# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
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
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            ii = list(range(i * B, i * B + B))
            W = W + alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            if ((t*(n/B)+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
    return res
    # TODO students should implement this

def plot_function(errors_tr,errors_te):
    errors = [errors_tr,errors_te]
    iterations = errors_tr.shape[1]
    for i in range(2):
        for j in range(errors_tr.shape[0]):
            matplotlib.pyplot.plot(range(iterations), errors[i][j,:])
        matplotlib.pyplot.figure(figsize=(8, 6))
        matplotlib.pyplot.legend(["Training error","Test error"])
        matplotlib.pyplot.title("Training vs Test errors")
        matplotlib.pyplot.xlabel("Iteration (model version)")
        matplotlib.pyplot.ylabel("Error")
        matplotlib.pyplot.savefig("error_estimate_plot_{}.png".format(["train","test"][i]))
        matplotlib.pyplot.show()

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    gamma = 0.0001
    alpha12 = 0.001
    alpha34 = 0.05
    num_epoch = 1
    monitor_period_alg12 = 6000
    monitor_period_alg34 = 100
    batch_size = 60
    W0 = np.random.normal(0, 1, size=(c, d))
    alg1_w = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha12,num_epoch,monitor_period_alg12)
    alg2_w = sgd_sequential_scan(Xs_tr,Ys_tr,gamma,W0,alpha12,num_epoch,monitor_period_alg12)


    alg3_w = sgd_minibatch(Xs_tr,Ys_tr, gamma, W0, alpha34, batch_size, num_epoch, monitor_period_alg34)
    alg4_w = sgd_minibatch_sequential_scan(Xs_tr,Ys_tr, gamma, W0, alpha34, batch_size, num_epoch, monitor_period_alg34)

    errors_tr = np.zeros((4,10*num_epoch))
    errors_te = np.zeros((4,10*num_epoch))
    for i in tqdm(range(4)):
        errors_tr[i,:] = [multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in alg1_w]
        errors_te[i,:] = [multinomial_logreg_error(Xs_te,Ys_te, w) for w in alg1_w]

    plot_function(errors_tr,errors_te)

    test = multinomial_logreg_grad_i(Xs_tr,Ys_tr,[3,40,66],gamma,W0)
    pass
    # TODO add code to produce figures

    ####Part 2 Exploration

