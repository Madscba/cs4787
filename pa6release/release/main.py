#!/usr/bin/env python3
import os

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program (should be set to 1 for Parts 1 and 3)
import numpy as np

implicit_num_threads = 4
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

matplotlib.style.use("bmh")
import threading
import time

from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


# SOME UTILITY FUNCTIONS that you may find to be useful, from my PA3 implementation
# feel free to use your own implementation instead if you prefer
def multinomial_logreg_error(Xs, Ys, W):
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error

def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    WdotX = numpy.dot(W, Xs[:,ii])
    expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0))
    softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis = 0)
    return numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
# END UTILITY FUNCTIONS


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(4787)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset



# SGD + Momentum (adapt from Programming Assignment 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # TODO students should use their implementation from programming assignment 3
    # or adapt this version, which is from my own solution to programming assignment 3
    models = []
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            # if ((ibatch+1) % monitor_period == 0):
            #     models.append(W)
    # return models
    return W

# SGD + Momentum (No Allocation) => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    W = numpy.zeros(W0.shape)
    V = numpy.zeros(W0.shape)

    # cb1, cb2, b1, cd1,cd2

    cb1 = numpy.zeros((c,B))
    cb2 = numpy.zeros((c, B))
    b1 = numpy.zeros((B))
    cd1 = numpy.zeros((c, d))
    cd2 = numpy.zeros((c, d))

    # intermediate_matrix2 = numpy.zeros((c, d))
    # XS_slice = numpy.ascontiguousarray(Xs[:, ii])


    Xs_splits = []
    Ys_splits = []
    for i in range( int(n/B)):
        Xs_splits.append(numpy.ascontiguousarray(Xs[:,i*B:i*B+B]))
        Ys_splits.append(Ys[:,i*B:i*B+B])

    # Xs_split = [numpy.ascontiguousarray(Xs[:, range(B)]) for ]
    # XS_slice = numpy.ascontiguousarray(numpy.zeros(Xs[:, range(B)].shape))


    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            XS_split = Xs_splits[ibatch]
            Ys_split = Ys_splits[ibatch]


            numpy.dot(W, XS_split, out=cb1)  # (c,d) x (d,b) = (c,b) cb1
            numpy.amax(cb1, axis=0,out=b1) # (c,b) cb2
            numpy.subtract(cb1, b1,out=cb1) #(c,b) cb1
            numpy.exp(cb1,out=cb1) #(c,b) cb1
            numpy.sum(cb1, axis=0,out=b1) # vector (b) (columns) b1
            numpy.divide(cb1, b1,out=cb1)  #(c,b) cb1
            numpy.subtract(cb1, Ys_split,out=cb1) #(c,b) cb1
            numpy.multiply(gamma, W,out=cd1)  #(c,d) cd1
            numpy.dot(cb1, XS_split.transpose(),out=cd2) #(c,b) x (b,d) = (c,d) cd2
            numpy.divide(cd2,B,out=cd2) #(c,d) cd2
            numpy.add(cd2, cd1,out=cd1) # (c,d)  cd1
            numpy.multiply(beta,V,out=cd2) #(c, d)  cd2
            numpy.multiply(alpha, cd1,out=cd1) # (c,d)  cd1
            numpy.subtract(cd2,cd1,out=V) #(c,d) V
            numpy.add(W,V,out=W) #(c,d)  W


            # WdotX = numpy.dot(W, XS_split,out=cb1) # (c,d) x (d,b) = (c,b) cb1
            # int2 = numpy.amax(WdotX, axis=0,out=cb2) # (c,b) cb2
            # int3 = numpy.subtract(WdotX, int2,out=cb1) #(c,b) cb1
            # expWdotX = numpy.exp(int3,out=cb1) #(c,b) cb1
            # int4 = numpy.sum(expWdotX, axis=0,out=b1) # vector (b) (columns) b1
            # softmaxWdotX = numpy.divide(expWdotX, int4,out=cb1)  #(c,b) cb1
            # int5 = numpy.subtract(softmaxWdotX, Ys_split,out=cb1) #(c,b) cb1
            # int6 = numpy.multiply(gamma, W,out=cd1)  #(c,d) cd1
            # int7 = numpy.dot(int5, XS_split.transpose(),out=cd2) #(c,b) x (b,d) = (c,d) cd2
            # int8 = numpy.divide(int7,B,out=cd2) #(c,d) cd2
            # multi_log_grad_i = numpy.add(int8, int6,out=cd1) # (c,d)  cd1
            # int9 = numpy.multiply(beta,V,out=cd2) #(c, d)  cd2
            # int10 = numpy.multiply(alpha, multi_log_grad_i,out=cd1) # (c,d)  cd1
            # V = numpy.subtract(int9,int10,out=V) #(c,d) V
            # W = numpy.add(W,V,out=W) #(c,d)  W
    return W


# SGD + Momentum (threaded)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO perform any global setup/initialization/allocation (students should implement this)

    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                # ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
                iter_barrier.wait()
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W


# SGD + Momentum (No Allocation) in 32-bits => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code


# SGD + Momentum (threaded, float32)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code

def plot_time(t1s,t2s,B_values,cores=1):
    pyplot.figure(figsize=(12, 8))
    pyplot.scatter(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t1s, label="t1")
    pyplot.scatter(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t2s, label="t2",marker='x')
    pyplot.plot(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t1s)
    pyplot.plot(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t2s)
    pyplot.title("Time comparison of preallocation")
    pyplot.xticks(np.linspace(np.min(B_values),np.max(B_values),len(B_values)),B_values)
    pyplot.xlabel("B_values")
    pyplot.ylabel("Time used (s)")
    pyplot.legend()
    pyplot.savefig(f"runtime_part_cores{cores}")
    pyplot.plot()
    # pyplot.show()
    # pyplot.clf()

def part1(Xs_tr, Ys_tr, Xs_te, Ys_te):
    (d, n) = Xs_tr.shape
    (c, n) = Ys_tr.shape
    alpha = 0.1
    beta = 0.9
    gamma = 0.0001
    num_epochs = 20
    W0 = numpy.random.rand(c,d)
    B_values = [8,16,30,64,200,600,3000]
    # B_values = [8,16]
    t1s = []
    t2s = []
    for B_size in B_values:
        B = B_size
        t1 = time.time()
        model1_w = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        t1 = time.time() - t1
        print("time for non-preallocated:",t1)

        t2 = time.time()
        model2_w = sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        t2 = time.time() - t2
        print("time for preallocated:",t2)

        err1 = multinomial_logreg_error(Xs_te,Ys_te,model1_w)
        err2 = multinomial_logreg_error(Xs_te,Ys_te,model2_w)

        print("error1: ",err1, ". error2 ",err2)

        t1s.append(t1)
        t2s.append(t2)

    plot_time(t1s,t2s,B_values,cores=implicit_num_threads)




if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    part1(Xs_tr, Ys_tr, Xs_te, Ys_te)

    # TODO add code to produce figures
