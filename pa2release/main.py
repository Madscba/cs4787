import os
import numpy as np
import random
import scipy
import matplotlib
import itertools
import mnist
import pickle
matplotlib.use('TkAgg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
from scipy.special import softmax
from tqdm import tqdm
import copy,time

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
    y = Ys[:, ii] #choose the corresponding targets

    softmax_input = np.matmul(W,x)
    un_reg_grad = np.matmul((softmax(softmax_input,axis=0)-y),x.T)
    l2_reg_grad = gamma * W
    return (un_reg_grad ) / len(ii) + l2_reg_grad #Average over the ii samples


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
    # TODO students should implement this
    _, n = Xs.shape
    num_epochs_corrected = num_epochs * n
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs_corrected)):
        if (t % monitor_period == 0):
            res.append(copy.deepcopy(W))
        datapoint_idx = random.sample(list(range(n)),1)
        W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, datapoint_idx, gamma, W)
    res.append(copy.deepcopy(W))
    return res


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
    # TODO students should implement this
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs)):
        for i in range(n):
            if ((t*n+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
            W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W)
    res.append(copy.deepcopy(W))       
    return res


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
    # TODO students should implement this
    _, n = Xs.shape
    num_epochs_corrected = int(num_epochs * (n/B)) #10*60000/60 = 10000
    W = copy.deepcopy(W0)
    res = []
    for t in tqdm(range(num_epochs_corrected)):
        if (t % monitor_period == 0):
            res.append(copy.deepcopy(W))
        datapoint_idx = random.sample(list(range(n)),B)
        tempW = np.zeros(W.shape)
        for i in datapoint_idx:
            tempW += multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W)
        W = W - alpha*tempW/B
        # W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, datapoint_idx, gamma, W)
    res.append(copy.deepcopy(W))
    return res



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
            if ((t*(n/B)+i) % monitor_period == 0):
                res.append(copy.deepcopy(W))
            tempW = np.zeros(W.shape)
            for i in ii:
                tempW += multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W)
            W = W - alpha*tempW/B
            # W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(copy.deepcopy(W))
    return res
    # TODO students should implement this
    
def plotSGDAlphaVariationsTr(sgd_alpha_variation_tr_errors, metadata):
    n, mp_alg12, n_epoch = metadata['n'], metadata['monitor_period_sgd'], metadata['num_epoch']
    alpha_sgd= metadata['alpha_sgd'],
    alpha_e1, alpha_e2 = metadata['alpha_exploration'][0], metadata['alpha_exploration'][1]
    sgd_ver1_tr_err, sgd_ver2_tr_err, sgd_ver3_tr_err = sgd_alpha_variation_tr_errors
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = sgd_alpha_variation_tr_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    matplotlib.pyplot.plot(X, sgd_ver1_tr_err)
    matplotlib.pyplot.plot(X, sgd_ver2_tr_err)
    matplotlib.pyplot.plot(X, sgd_ver3_tr_err)
    matplotlib.pyplot.title("Training Errors of SGD with Varying Learning Rates")
    matplotlib.pyplot.legend(["Training Error w/ alpha={}".format(alpha_sgd),
                              "Training error w/ alpha={}".format(alpha_e1),
                              "Training error w/ alpha={}".format(alpha_e2)])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_tr_error_w_diff_alphas.png")
    matplotlib.pyplot.show()

def plotSGDAlphaVariationsTe(sgd_alpha_variation_te_errors, metadata):
    n, mp_alg12, n_epoch = metadata['n'], metadata['monitor_period_sgd'], metadata['num_epoch']
    alpha_sgd= metadata['alpha_sgd'],
    alpha_e1, alpha_e2 = metadata['alpha_exploration'][0], metadata['alpha_exploration'][1]
    sgd_ver1_te_err, sgd_ver2_te_err, sgd_ver3_te_err = sgd_alpha_variation_te_errors
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = sgd_alpha_variation_te_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    matplotlib.pyplot.plot(X, sgd_ver1_te_err)
    matplotlib.pyplot.plot(X, sgd_ver2_te_err)
    matplotlib.pyplot.plot(X, sgd_ver3_te_err)
    matplotlib.pyplot.title("Testing Errors of SGD with Varying Learning Rates")
    matplotlib.pyplot.legend(["Testing Error w/ alpha={}".format(alpha_sgd),
                              "Testing error w/ alpha={}".format(alpha_e1),
                              "Testing error w/ alpha={}".format(alpha_e2)])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_te_error_w_diff_alphas.png")
    matplotlib.pyplot.show()

def plotMBSGDVariationsTr(mb_sgd_variation_tr_errors, metadata):
    n, monitor_period_mb_sgd, n_epoch, batch_size = metadata['n'], metadata['monitor_period_mb_sgd'], metadata['num_epoch'], metadata['batch_size']
    alpha_mb_sgd, alpha_exploration =  metadata['alpha_mb_sgd'], metadata['alpha_exploration']
    batch_size_mb_sgd, batch_size_and_monitor_freq_exploration = metadata['batch_size'], metadata['batch_size_and_monitor_freq_exploration']
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = mb_sgd_variation_tr_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    
    for mb_sgd_variation_tr_err in mb_sgd_variation_tr_errors:
        matplotlib.pyplot.plot(X, mb_sgd_variation_tr_err)
    matplotlib.pyplot.title("Training Errors of Mini-Batch SGD with Varying Learning Rates and Batch Sizes")
    matplotlib.pyplot.legend(["Training Error w/ alpha={} and batch size={}".format(alpha_mb_sgd,         batch_size_mb_sgd),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[2], batch_size_mb_sgd),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[3], batch_size_mb_sgd),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[2], batch_size_and_monitor_freq_exploration[0][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[2], batch_size_and_monitor_freq_exploration[1][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[3], batch_size_and_monitor_freq_exploration[0][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[3], batch_size_and_monitor_freq_exploration[1][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[4], batch_size_and_monitor_freq_exploration[1][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[4], batch_size_and_monitor_freq_exploration[0][0]),
                              "Training error w/ alpha={} and batch size={}".format(alpha_exploration[5], batch_size_and_monitor_freq_exploration[0][0]),
                              ])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_mb_sgd_tr_error_w_diff_alphas_and_batch_sizes.png")
    matplotlib.pyplot.show()

def plotMBSGDVariationsTe(mb_sgd_variation_te_errors, metadata):
    n, monitor_period_mb_sgd, n_epoch, batch_size = metadata['n'], metadata['monitor_period_mb_sgd'], metadata['num_epoch'], metadata['batch_size']
    alpha_mb_sgd, alpha_exploration =  metadata['alpha_mb_sgd'], metadata['alpha_exploration']
    batch_size_mb_sgd, batch_size_and_monitor_freq_exploration = metadata['batch_size'], metadata['batch_size_and_monitor_freq_exploration']
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = mb_sgd_variation_te_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    
    print(X)
    for mb_sgd_variation_te_err in mb_sgd_variation_te_errors:
        matplotlib.pyplot.plot(X, mb_sgd_variation_te_err)
    matplotlib.pyplot.title("Testing Errors of Mini-Batch SGD with Varying Learning Rates ahd Batch Sizes")
    matplotlib.pyplot.legend(["Testing Error w/ alpha={} and batch size={}".format(alpha_mb_sgd,         batch_size_mb_sgd),
                              "Testing error w/ alpha={} and batch size={}".format(alpha_exploration[3], batch_size_mb_sgd),
                              "Testing error w/ alpha={} and batch size={}".format(alpha_exploration[4], batch_size_and_monitor_freq_exploration[0][0]),
                              ])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_mb_sgd_te_error_w_diff_alphas_and_batch_sizes.png")
    matplotlib.pyplot.show()

def plotVariationsTr(variation_tr_errors, metadata):
    n, monitor_period_mb_sgd, n_epoch, batch_size = metadata['n'], metadata['monitor_period_mb_sgd'], metadata['num_epoch'], metadata['batch_size']
    alpha_mb_sgd, alpha_exploration = metadata['alpha_mb_sgd'], metadata['alpha_exploration']
    batch_size_mb_sgd, batch_size_and_monitor_freq_exploration = metadata['batch_size'], metadata['batch_size_and_monitor_freq_exploration']
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = variation_tr_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    
    for mb_sgd_variation_tr_err in variation_tr_errors:
        matplotlib.pyplot.plot(X, mb_sgd_variation_tr_err)
    matplotlib.pyplot.title("Training Errors of SGD and Mini-Batch SGD with Varying Learning Rates ahd Batch Sizes")
    matplotlib.pyplot.legend(["Training error of SGD w/ alpha={}".format(alpha_exploration[0]),
                              "Training error of SGD w/ alpha={}".format(alpha_exploration[1]),
                              "Training error w/ MB_SGD alpha={} and batch size={}".format(alpha_exploration[3], batch_size),
                              ])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_5_tr.png")
    matplotlib.pyplot.show()


def plotVariationsTe(variation_te_errors, metadata):
    n, monitor_period_mb_sgd, n_epoch, batch_size = metadata['n'], metadata['monitor_period_mb_sgd'], metadata['num_epoch'], metadata['batch_size']
    alpha_mb_sgd, alpha_exploration = metadata['alpha_mb_sgd'], metadata['alpha_exploration']
    batch_size_mb_sgd, batch_size_and_monitor_freq_exploration = metadata['batch_size'], metadata['batch_size_and_monitor_freq_exploration']
    matplotlib.pyplot.figure(figsize=(8, 6))
    iterations = variation_te_errors[0].shape[0]
    X = np.linspace(0,n_epoch-.1,num=iterations)
    
    for mb_sgd_variation_te_err in variation_te_errors:
        matplotlib.pyplot.plot(X, mb_sgd_variation_te_err)
    matplotlib.pyplot.title("Testing Errors of SGD and Mini-Batch SGD with Varying Learning Rates and Batch Sizes")
    matplotlib.pyplot.legend(["Testing error of SGD w/ alpha={}".format(alpha_exploration[0]),
                              "Testing error of SGD w/ alpha={}".format(alpha_exploration[1]),
                              "Testing error w/ MB_SGD alpha={} and batch size={}".format(alpha_exploration[3], batch_size),
                              ])
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("pt2_5_te.png")
    matplotlib.pyplot.show()
# ___________________________________Part 1_____________________________________________________
def implementation(Xs_tr, Ys_tr, Xs_te, Ys_te, metadata):
    # Hyperparameters
    gamma, W0, num_epoch = metadata['gamma'], metadata['W0'], metadata['num_epoch']
    monitor_period_sgd, monitor_period_mb_sgd = metadata['monitor_period_sgd'], metadata['monitor_period_mb_sgd']
    alpha_sgd, alpha_mb_sgd = metadata['alpha_sgd'], metadata['alpha_mb_sgd']
    # SGD
    sgd = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_sgd,num_epoch,monitor_period_sgd)
    sgd_seq = sgd_sequential_scan(Xs_tr,Ys_tr,gamma,W0,alpha_sgd,num_epoch,monitor_period_sgd)
    
    # Minibatch SGD
    mb_sgd = sgd_minibatch(Xs_tr,Ys_tr, gamma, W0, alpha_mb_sgd, batch_size, num_epoch, monitor_period_mb_sgd)
    mb_sgd_seq = sgd_minibatch_sequential_scan(Xs_tr,Ys_tr, gamma, W0, alpha_mb_sgd, batch_size, num_epoch, monitor_period_mb_sgd)

    variations = [sgd, sgd_seq, mb_sgd, mb_sgd_seq]
    n, m = sgd.shape
    
    errors_tr = np.zeros((4, n))
    errors_te = np.zeros((4, n))
    for i in tqdm(range(len(variations))):
        errors_tr[i, :] = [multinomial_logreg_error(Xs_tr,Ys_tr, weights) for weights in variations[i][:,1]]
        errors_te[i, :] = [multinomial_logreg_error(Xs_te,Ys_te, weights) for weights in variations[i][:,1]]

    errors = [errors_tr,errors_te]
    iterations = errors_tr.shape[1]
    epochs = iterations/10
    # Plot Training and Test Errors of SGD Variations
    plot_markers = ['o','+','v','x']
    for i in range(2):
        matplotlib.pyplot.close()
        matplotlib.pyplot.annotate
        matplotlib.pyplot.figure(figsize=(9, 7))
        matplotlib.pyplot.title("{}".format(["Training", "Test errors"][i]))
        matplotlib.pyplot.xlabel("Number of epochs (model version)")
        matplotlib.pyplot.ylabel("Error")

        for j in range(errors_tr.shape[0]):
            matplotlib.pyplot.plot(np.linspace(0,epochs-.1,num=iterations), errors[i][j,:],marker=plot_markers[j],markersize=6,linestyle="--")
        matplotlib.pyplot.legend(["SGD_random_sampling, final error: {:.3f}".format(errors[i][0,-1]), "SGD_sequential_sampling, final error: {:.3f}".format(errors[i][1,-1]), "Minibatch_SGD_random_sampling, final error: {:.3f}".format(errors[i][2,-1]), "Mini_batch_SGD_sequential_sampling, final error: {:.3f}".format(errors[i][3,-1])])
        matplotlib.pyplot.savefig("error_estimate_plot_{}.png".format(["train","test"][i]))
        matplotlib.pyplot.show()
    return variations
    # test = multinomial_logreg_grad_i(Xs_tr,Ys_tr,[3,40,66],gamma,W0)
    
# ___________________________________Part 2_____________________________________________________
def exploration(Xs_tr,Ys_tr, Xs_te,Ys_te, metadata):
    print("### Exploration ###")
    gamma, W0, num_epoch = metadata['gamma'], metadata['W0'], metadata['num_epoch']
    monitor_period_sgd, monitor_period_mb_sgd = metadata['monitor_period_sgd'], metadata['monitor_period_mb_sgd']
    alpha_sgd, alpha_mb_sgd = metadata['alpha_sgd'], metadata['alpha_mb_sgd']
    alpha_exploration = [0.005, 0.01, 0.1, 0.5, 2, 5]
    batch_size_and_monitor_freq_exploration = [(600,10), (6000, 1)]
    metadata['batch_size_and_monitor_freq_exploration'] = batch_size_and_monitor_freq_exploration
    metadata['alpha_exploration'] = alpha_exploration
    # sgd_ver1 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_sgd,num_epoch,monitor_period_sgd)
    # try:
    #     sgd_ver1 = np.genfromtxt("sgd_ver1.csv", delimiter=",")
    # except:
    #     sgd_ver1 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_sgd,num_epoch,monitor_period_sgd)
    #     np.savetxt("sgd_ver1.csv", sgd_ver1, delimiter=",")
    # try:
    #     sgd_ver1_tr_error = np.loadtxt("sgd_ver1_tr_error.csv", delimiter=",")
    # except:
    #     sgd_ver1_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in sgd_ver1])
    #     np.savetxt("sgd_ver1_tr_error.csv", sgd_ver1_tr_error, delimiter=",")

    # sgd_ver2 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_exploration[0],num_epoch,monitor_period_sgd)
    # # try:
    # #     sgd_ver2 = np.genfromtxt("sgd_ver2.csv", delimiter=",")
    # # except:
    # #     sgd_ver2 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_exploration[0],num_epoch,monitor_period_sgd)
    # #     np.savetxt("sgd_ver2.csv", sgd_ver2, delimiter=",")
    # try:
    #     sgd_ver2_tr_error = np.loadtxt("sgd_ver2_tr_error.csv", delimiter=",")
    # except:
    #     sgd_ver2_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in sgd_ver2])
    #     np.savetxt("sgd_ver2_tr_error.csv", sgd_ver2_tr_error, delimiter=",")
    
    # sgd_ver3 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_exploration[1],num_epoch,monitor_period_sgd)
    # # try:
    # #     sgd_ver3 = np.loadtxt("sgd_ver3.csv", delimiter=",")
    # # except:
    # #     sgd_ver3 = stochastic_gradient_descent(Xs_tr,Ys_tr,gamma,W0,alpha_exploration[1],num_epoch,monitor_period_sgd)
    # #     np.savetxt("sgd_ver3.csv", sgd_ver3, delimiter=",")
    # try:
    #     sgd_ver3_tr_error = np.loadtxt("sgd_ver3_tr_error.csv", delimiter=",")
    # except:
    #     sgd_ver3_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in sgd_ver3])
    #     np.savetxt("sgd_ver3_tr_error.csv", sgd_ver3_tr_error, delimiter=",")
    # sgd_alpha_variation_tr_errors = [sgd_ver1_tr_error, sgd_ver2_tr_error, sgd_ver3_tr_error]
    # # plotSGDAlphaVariationsTr(sgd_alpha_variation_tr_errors, metadata)
    # print("Training Error after 5 epochs with alpha 0.05:", sgd_ver1_tr_error[50])
    # print("Training Error after 5 epochs with alpha 0.1:",  sgd_ver2_tr_error[50])
    # print("Training Error after 5 epochs with alpha 0.5:",  sgd_ver3_tr_error[50])
    # print("Final training error with alpha 0.001:", sgd_ver1_tr_error[-1])
    # print("Final training error with alpha 0.005:", sgd_ver2_tr_error[-1])
    # print("Final training error with alpha 0.01:" , sgd_ver3_tr_error[-1])

    # try:
    #     sgd_ver1_te_error = np.loadtxt("sgd_ver1_te_error.csv", delimiter=",")
    # except:
    #     sgd_ver1_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in sgd_ver1[:, 1]])
    #     np.savetxt("sgd_ver1_te_error.csv", sgd_ver1_te_error, delimiter=",")
    try:
        sgd_ver2_te_error = np.loadtxt("sgd_ver2_te_error.csv", delimiter=",")
    except:
        sgd_ver2_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in sgd_ver2])
        np.savetxt("sgd_ver2_te_error.csv", sgd_ver2_te_error, delimiter=",")
    try:
        sgd_ver3_te_error = np.loadtxt("sgd_ver3_te_error.csv", delimiter=",")
    except:
        sgd_ver3_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in sgd_ver3])
        np.savetxt("sgd_ver3_te_error.csv", sgd_ver3_te_error, delimiter=",")
    # sgd_alpha_variation_te_errors = [sgd_ver1_te_error, sgd_ver2_te_error, sgd_ver3_te_error]
    # plotSGDAlphaVariationsTe(sgd_alpha_variation_te_errors, metadata)
    # print("Testing Error after 5 epochs with alpha 0.05:", sgd_ver1_te_error[50])
    # print("Testing Error after 5 epochs with alpha 0.1:",  sgd_ver2_te_error[50])
    # print("Testing Error after 5 epochs with alpha 0.5:",  sgd_ver3_te_error[50])
    # print("Final testing error with alpha 0.001:", sgd_ver1_te_error[-1])
    # print("Final testing error with alpha 0.005:", sgd_ver2_te_error[-1])
    # print("Final testing error with alpha 0.01:" , sgd_ver3_te_error[-1])
    

    # alpha = 0.05, batch_size = 60 or Part1 Minibatch Sequential SGD 
    try:
        mb_sgd_ver1_tr_error = np.loadtxt("mb_sgd_ver1_tr_error.csv", delimiter=",")
    except:
        mb_sgd_ver1 = sgd_minibatch_sequential_scan(Xs_tr,Ys_tr, gamma, W0, alpha_mb_sgd, batch_size, num_epoch, monitor_period_mb_sgd)
        mb_sgd_ver1_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in mb_sgd_ver1[:, 1]])
        np.savetxt("mb_sgd_ver1_tr_error.csv", mb_sgd_ver1_tr_error, delimiter=",")
    # alpha = 0.5, batch_size = 60 
    try:
        mb_sgd_ver3_tr_error = np.loadtxt("mb_sgd_ver3_tr_error.csv", delimiter=",")
    except:
        mb_sgd_ver3 = sgd_minibatch_sequential_scan(Xs_tr,Ys_tr, gamma, W0, alpha_exploration[3], batch_size, num_epoch, monitor_period_mb_sgd)
        mb_sgd_ver3_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in mb_sgd_ver3[:, 1]])
        np.savetxt("mb_sgd_ver3_tr_error.csv", mb_sgd_ver3_tr_error, delimiter=",")
    # custom_batch_size, custom_monitor_period = batch_size_and_monitor_freq_exploration[0]
    try:
        mb_sgd_ver9_tr_error = np.loadtxt("mb_sgd_ver9_tr_error.csv", delimiter=",")
    except:
        mb_sgd_ver9 = sgd_minibatch_sequential_scan(Xs_tr,Ys_tr, gamma, W0, alpha_exploration[4], custom_batch_size, num_epoch, custom_monitor_period)
        mb_sgd_ver9_tr_error = np.array([multinomial_logreg_error(Xs_tr,Ys_tr, w) for w in mb_sgd_ver9[:, 1]])
        np.savetxt("mb_sgd_ver9_tr_error.csv", mb_sgd_ver9_tr_error, delimiter=",")

    # mb_sgd_variation_tr_errors = [mb_sgd_ver1_tr_error, mb_sgd_ver3_tr_error, mb_sgd_ver9_tr_error]
    # plotMBSGDVariationsTr(mb_sgd_variation_tr_errors, metadata)
    # print("Training Error after 5 epochs with alpha = 0.05, batch_size = 60:", mb_sgd_ver1_tr_error[50])
    # print("Training Error after 5 epochs with alpha = 0.5, batch_size = 60:" , mb_sgd_ver3_tr_error[50])
    # print("Training Error after 5 epochs with alpha = 2, batch_size = 600:" , mb_sgd_ver9_tr_error[50])

    # print("Final training error with alpha = 0.05, batch_size = 60:", mb_sgd_ver1_tr_error[-1])
    # print("Final training error with alpha = 0.5, batch_size = 60:" , mb_sgd_ver3_tr_error[-1])
    # print("Final training error with alpha = 2, batch_size = 600:" , mb_sgd_ver9_tr_error[-1])
    # __________________________ MB SGD on Test Data ________________________________
    # Baseline
    try:
        mb_sgd_ver1_te_error = np.loadtxt("mb_sgd_ver1_te_error.csv", delimiter=",")
    except:
        mb_sgd_ver1_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in mb_sgd_ver1[:, 1]])
        np.savetxt("mb_sgd_ver1_te_error.csv", mb_sgd_ver1_te_error, delimiter=",")
    # alpha = 0.5, batch_size = 60 
    try:
        mb_sgd_ver3_te_error = np.loadtxt("mb_sgd_ver3_te_error.csv", delimiter=",")
    except:
        mb_sgd_ver3_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in mb_sgd_ver3[:, 1]])
        np.savetxt("mb_sgd_ver3_te_error.csv", mb_sgd_ver3_te_error, delimiter=",")
    # alpha = 2, batch_size = 600
    try:
        mb_sgd_ver9_te_error = np.loadtxt("mb_sgd_ver9_te_error.csv", delimiter=",")
    except:
        mb_sgd_ver9_te_error = np.array([multinomial_logreg_error(Xs_te,Ys_te, w) for w in mb_sgd_ver9[:, 1]])
        np.savetxt("mb_sgd_ver9_te_error.csv", mb_sgd_ver9_te_error, delimiter=",")
    mb_sgd_variation_te_errors = [mb_sgd_ver1_te_error, mb_sgd_ver3_te_error, mb_sgd_ver9_te_error]
    # plotMBSGDVariationsTe(mb_sgd_variation_te_errors, metadata)
    # print("Testing Error after 5 epochs with alpha = 0.05, batch_size = 60:", mb_sgd_ver1_te_error[50])
    # print("Testing Error after 5 epochs with alpha = 2, batch_size = 600:" , mb_sgd_ver3_te_error[50])
    # print("Testing Error after 5 epochs with alpha = 5, batch_size = 600:" , mb_sgd_ver9_te_error[50])

    # print("Final testing error with alpha = 0.05, batch_size = 60:", mb_sgd_ver1_te_error[-1])
    # print("Final testing error with alpha = 0.1, batch_size = 60:" , mb_sgd_ver3_te_error[-1])
    # print("Final testing error with alpha = 0.5, batch_size = 60:" , mb_sgd_ver9_te_error[-1])
    
    # ______________________ PART 2.5 __________________________
    # variation_tr_errors = [sgd_ver2_tr_error, sgd_ver3_tr_error, mb_sgd_ver3_tr_error]
    # plotVariationsTr(variation_tr_errors, metadata)
    variation_te_errors = [sgd_ver2_te_error, sgd_ver3_te_error, mb_sgd_ver3_te_error]
    plotVariationsTe(variation_te_errors, metadata)

def system_evaluation(metadata):
    print("### System_Evalualtion ###")
    iterations = 10
    t_1 = time.time()
    for _ in tqdm(range(iterations)):
        stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, metadata['alpha_sgd'], num_epoch, metadata['monitor_period_sgd'])
    t_1 = time.time() - t_1

    print(t_1/iterations)

    # t_2 = time.time()
    # for _ in tqdm(range(iterations)):
    #     sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha12, num_epoch, monitor_period_alg12)
    # t_2 = time.time() - t_2
    #
    # t_3 = time.time()
    # for _ in tqdm(range(iterations)):
    #     sgd_minibatch(Xs_tr, Ys_tr, gamma, W0, alpha34, batch_size, num_epoch, monitor_period_alg34)
    # t_3 = time.time() - t_3
    #
    # t_4 = time.time()
    # for _ in tqdm(range(iterations)):
    #     sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha34, batch_size, num_epoch,monitor_period_alg34)
    # t_4 = time.time() - t_4
    # print("Time for algo2:{}.\n Time for algo3:{}.\n Time for algo4:{}".format(t_2/iterations,t_3/iterations,t_4/iterations))
    # return [t_2,t_3,t_4]
    # print("Time for algo1:{}.\n Time for algo2:{}.\n Time for algo3:{}.\n Time for algo4:{}".format(t_1/iterations,t_2/iterations,t_3/iterations,t_4/iterations))
    # return [t_1,t_2,t_3,t_4]
if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    gamma = 0.0001
    alpha_sgd = 0.001
    alpha_mb_sgd = 0.05
    num_epoch = 10
    monitor_period_sgd = 6000
    monitor_period_mb_sgd = 100
    batch_size = 60
    W0 = np.random.normal(0, 1, size=(c, d))
    metadata = {'d': d, 
                'n' : n, 
                'c' : c, 
                'gamma' : gamma, 
                'alpha_sgd' : alpha_sgd, 
                'alpha_mb_sgd' : alpha_mb_sgd,
                'num_epoch' : num_epoch,
                'monitor_period_sgd' : monitor_period_sgd,
                'monitor_period_mb_sgd' : monitor_period_mb_sgd,
                'W0' : W0,
                'batch_size': batch_size}
    # [sgd, sgd_seq, mb_sgd, mb_sgd_seq] = implementation(Xs_tr, Ys_tr, Xs_te, Ys_te, metadata)
    

    ####Part 2 Exploration
    exploration(Xs_tr, Ys_tr, Xs_te, Ys_te, metadata)


    ####Part 3 System Evaluation
    # system_evaluation(metadata)
    # a = 2
