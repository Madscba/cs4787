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
    PICKLE_FILE = "C:\\Users\\hyun0\\Documents\\CS4787\\cs4787\\pa3release\\data\\MNIST.pickle"
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
    W = numpy.zeros(W0.shape)
    V = numpy.zeros(W0.shape)
    B_prime = int(B / num_threads)
    gradients = numpy.zeros((num_threads, W0.shape[0], W0.shape[1]))
    sum_gradients = numpy.zeros(W0.shape)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    Xs_splits = []
    Ys_splits = []
    # n = 256 = 2^8
    # B = 16 = 2^4
    # num_threads = 4

    for i in range(int(n / B * num_threads)):
        Xs_splits.append(numpy.ascontiguousarray(Xs[:, i * B_prime:i * B_prime + B_prime]))
        Ys_splits.append(Ys[:, i * B_prime:i * B_prime + B_prime])

    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations

        cb1 = numpy.zeros((c, B_prime))
        b1 = numpy.zeros((B_prime))
        cd1 = numpy.zeros((c, d))
        cd2 = numpy.zeros((c, d))
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                Xs_split = Xs_splits[ibatch*num_threads+ithread]
                Ys_split = Ys_splits[ibatch*num_threads+ithread]

                numpy.dot(W, Xs_split, out=cb1)  # (c,d) x (d,b) = (c,b) cb1
                numpy.amax(cb1, axis=0, out=b1)  # (c,b) cb2
                numpy.subtract(cb1, b1, out=cb1)  # (c,b) cb1
                numpy.exp(cb1, out=cb1)  # (c,b) cb1
                numpy.sum(cb1, axis=0, out=b1)  # vector (b) (columns) b1
                numpy.divide(cb1, b1, out=cb1)  # (c,b) cb1
                numpy.subtract(cb1, Ys_split, out=cb1)  # (c,b) cb1
                numpy.multiply(gamma, W, out=cd1)  # (c,d) cd1
                numpy.dot(cb1, Xs_split.transpose(), out=cd2)  # (c,b) x (b,d) = (c,d) cd2
                numpy.divide(cd2, B, out=cd2)  # (c,d) cd2
                numpy.add(cd2, cd1, out=cd1)  # (c,d)  cd1
                gradients[ithread, :, :] = cd1
                iter_barrier.wait()
                # Wait until new global gradient has been calculated
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        # print("running thread ", t)
        t.start()

    print("Running threaded minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            numpy.sum(gradients, axis=0, out=sum_gradients)
            # update V
            numpy.multiply(beta, V, out=V)
            numpy.multiply(alpha, sum_gradients, out=sum_gradients)
            numpy.subtract(V, sum_gradients, out=V)
            # update W
            numpy.add(W, V, out=W)
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
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    W = numpy.zeros(W0.shape, dtype=numpy.float32)
    V = numpy.zeros(W0.shape, dtype=numpy.float32)

    # cb1, cb2, b1, cd1,cd2

    cb1 = numpy.zeros((c, B), dtype=numpy.float32)
    b1 = numpy.zeros((B), dtype=numpy.float32)
    cd1 = numpy.zeros((c, d), dtype=numpy.float32)
    cd2 = numpy.zeros((c, d), dtype=numpy.float32)

    # intermediate_matrix2 = numpy.zeros((c, d))
    # XS_slice = numpy.ascontiguousarray(Xs[:, ii])

    Xs_splits = []
    Ys_splits = []
    for i in range(int(n / B)):
        Xs_splits.append(numpy.ascontiguousarray(Xs[:, i * B:i * B + B]))
        Ys_splits.append(Ys[:, i * B:i * B + B])

    print("Running minibatch sequential-scan SGD with momentum (no allocation) and float32")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n / B)):
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            XS_split = Xs_splits[ibatch].astype(numpy.float32)
            Ys_split = Ys_splits[ibatch].astype(numpy.float32)

            numpy.dot(W, XS_split, out=cb1)  # (c,d) x (d,b) = (c,b) cb1
            numpy.amax(cb1, axis=0, out=b1)  # (c,b) cb2
            numpy.subtract(cb1, b1, out=cb1)  # (c,b) cb1
            numpy.exp(cb1, out=cb1)  # (c,b) cb1
            numpy.sum(cb1, axis=0, out=b1)  # vector (b) (columns) b1
            numpy.divide(cb1, b1, out=cb1)  # (c,b) cb1
            numpy.subtract(cb1, Ys_split, out=cb1)  # (c,b) cb1
            numpy.multiply(gamma, W, out=cd1)  # (c,d) cd1
            numpy.dot(cb1, XS_split.transpose(), out=cd2)  # (c,b) x (b,d) = (c,d) cd2
            numpy.divide(cd2, B, out=cd2)  # (c,d) cd2
            numpy.add(cd2, cd1, out=cd1)  # (c,d)  cd1
            numpy.multiply(beta, V, out=cd2)  # (c, d)  cd2
            numpy.multiply(alpha, cd1, out=cd1)  # (c,d)  cd1
            numpy.subtract(cd2, cd1, out=V)  # (c,d) V
            numpy.add(W, V, out=W)  # (c,d)  W
    return W


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
    # TODO perform any global setup/initialization/allocation (students should implement this)
    W = numpy.zeros(W0.shape, dtype=numpy.float32)
    V = numpy.zeros(W0.shape, dtype=numpy.float32)
    B_prime = int(B / num_threads)
    gradients = numpy.zeros((num_threads, W0.shape[0], W0.shape[1]), dtype=numpy.float32)
    sum_gradients = numpy.zeros(W0.shape, dtype=numpy.float32)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    Xs_splits = []
    Ys_splits = []
    # n = 256 = 2^8
    # B = 16 = 2^4
    # num_threads = 4

    for i in range(int(n / B * num_threads)):
        Xs_splits.append(numpy.ascontiguousarray(Xs[:, i * B_prime:i * B_prime + B_prime]))
        Ys_splits.append(Ys[:, i * B_prime:i * B_prime + B_prime])

    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations

        cb1 = numpy.zeros((c, B_prime), dtype=numpy.float32)
        b1 = numpy.zeros((B_prime), dtype=numpy.float32)
        cd1 = numpy.zeros((c, d), dtype=numpy.float32)
        cd2 = numpy.zeros((c, d), dtype=numpy.float32)
        for it in range(num_epochs):
            for ibatch in range(int(n / B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                Xs_split = Xs_splits[ibatch * num_threads + ithread].astype(numpy.float32)
                Ys_split = Ys_splits[ibatch * num_threads + ithread].astype(numpy.float32)

                numpy.dot(W, Xs_split, out=cb1)  # (c,d) x (d,b) = (c,b) cb1
                numpy.amax(cb1, axis=0, out=b1)  # (c,b) cb2
                numpy.subtract(cb1, b1, out=cb1)  # (c,b) cb1
                numpy.exp(cb1, out=cb1)  # (c,b) cb1
                numpy.sum(cb1, axis=0, out=b1)  # vector (b) (columns) b1
                numpy.divide(cb1, b1, out=cb1)  # (c,b) cb1
                numpy.subtract(cb1, Ys_split, out=cb1)  # (c,b) cb1
                numpy.multiply(gamma, W, out=cd1)  # (c,d) cd1
                numpy.dot(cb1, Xs_split.transpose(), out=cd2)  # (c,b) x (b,d) = (c,d) cd2
                numpy.divide(cd2, B, out=cd2)  # (c,d) cd2
                numpy.add(cd2, cd1, out=cd1)  # (c,d)  cd1
                gradients[ithread, :, :] = cd1
                iter_barrier.wait()
                # Wait until new global gradient has been calculated
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        # print("running thread ", t)
        t.start()

    print("Running threaded minibatch sequential-scan SGD with momentum (%d threads) and float32" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n / B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            numpy.sum(gradients, axis=0, out=sum_gradients)
            # update V
            numpy.multiply(beta, V, out=V)
            numpy.multiply(alpha, sum_gradients, out=sum_gradients)
            numpy.subtract(V, sum_gradients, out=V)
            # update W
            numpy.add(W, V, out=W)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W

def plot_time(t1s,t2s,B_values,cores=1):
    pyplot.figure(figsize=(12, 8))
    pyplot.scatter(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t1s, label="t1")
    pyplot.scatter(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t2s, label="t2",marker='x')
    t1_plot = pyplot.plot(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t1s)
    t1_plot.set_label('w/o preallocation & w/o manual threading')
    t2_plot = pyplot.plot(np.linspace(np.min(B_values),np.max(B_values),len(B_values)), t2s)
    t2_plot.set_label('w/ preallocation & w/o manual threading')

    pyplot.title("Time comparison of preallocation")
    pyplot.xticks(np.linspace(np.min(B_values),np.max(B_values),len(B_values)),B_values)
    pyplot.xlabel("B_values")
    pyplot.ylabel("Time used (s)")
    pyplot.legend()
    pyplot.savefig(f"runtime_part_cores{cores}")
    pyplot.plot()
    # pyplot.show()
    # pyplot.clf()

def plot_time_pt2(t1s_single_core, t2s_single_core, t1s_multi_core, t2s_multi_core, B_values, cores=implicit_num_threads):
    pyplot.figure(figsize=(12, 8))
    t1_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_single_core)
    t1_plot.set_label(f"w/o preallocation w/ 1 core")
    t2_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_single_core)
    t2_plot.set_label(f"w/ preallocation w/ 1 core")
    t3_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_multi_core)
    t3_plot.set_label(f"w/o preallocation w/ 4 cores")
    t4_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_multi_core)
    t4_plot.set_label(f"w preallocation w/ 4 cores")
    pyplot.title("Time comparisons of preallocation and threading")
    pyplot.xticks(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), B_values)
    pyplot.xlabel("B_values")
    pyplot.ylabel("Time used (s)")
    pyplot.legend()
    pyplot.savefig("Impact of Preallocation and Number of Cores")
    pyplot.plot()
    # pyplot.show()
    # pyplot.clf()

def plot_time_pt3(ts_manual_threading, B_values, cores=1):
    pyplot.figure(figsize=(12, 8))
    t1s_single_core_filepath = 't1s_1_core.npy'
    t2s_single_core_filepath = 't2s_1_core.npy'
    t1s_multi_core_filepath = f't1s_4_core.npy'
    t2s_multi_core_filepath = f't2s_4_core.npy'
    t1s_single_core = numpy.load(t1s_single_core_filepath)
    t2s_single_core = numpy.load(t2s_single_core_filepath)
    t1s_multi_core = numpy.load(t1s_multi_core_filepath)
    t2s_multi_core = numpy.load(t2s_multi_core_filepath)
    t1_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_single_core)
    t1_plot.set_label(f"w/o preallocation w/ 1 core w/o manual threading")
    t2_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_single_core)
    t2_plot.set_label(f"w/ preallocation w/ 1 core w/o manual threading")
    t3_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_multi_core)
    t3_plot.set_label(f"w/o preallocation w/ 4 cores w/o manual threading")
    t4_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_multi_core)
    t4_plot.set_label(f"w preallocation w/ 4 cores w/o manual threading")
    t5_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), ts_manual_threading)
    t5_plot.set_label('w/ preallocation & w/ 1 core w/ manual threading')
    pyplot.title("Time comparisons of preallocation and threading")
    pyplot.xticks(np.linspace(np.min(B_values),np.max(B_values),len(B_values)),B_values)
    pyplot.xlabel("B_values")
    pyplot.ylabel("Time used (s)")
    pyplot.legend()
    pyplot.savefig("Impact of Preallocation and Threading")
    pyplot.plot()
    # pyplot.show()
    # pyplot.clf()


def part12(Xs_tr, Ys_tr, Xs_te, Ys_te):
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
    t1s = numpy.asarray(t1s)
    t2s = numpy.asarray(t2s)
    t1s_filepath = f't1s_{implicit_num_threads}_core.npy'
    t2s_filepath = f't2s_{implicit_num_threads}_core.npy'
    numpy.save(t1s_filepath, t1s)
    numpy.save(t2s_filepath, t2s)
    t1s_single_core_filepath = 't1s_1_core.npy'
    t2s_single_core_filepath = 't2s_1_core.npy'
    t1s_multi_core_filepath = f't1s_4_core.npy'
    t2s_multi_core_filepath = f't2s_4_core.npy'
    t1s_single_core = numpy.load(t1s_single_core_filepath)
    t2s_single_core = numpy.load(t2s_single_core_filepath)
    t1s_multi_core = numpy.load(t1s_multi_core_filepath)
    t2s_multi_core = numpy.load(t2s_multi_core_filepath)

    plot_time_pt2(t1s_single_core, t2s_single_core, t1s_multi_core, t2s_multi_core, B_values, 4)

def part3(Xs_tr, Ys_tr, Xs_te, Ys_te):
    (d, n) = Xs_tr.shape
    (c, n) = Ys_tr.shape
    alpha = 0.1
    beta = 0.9
    gamma = 0.0001
    num_epochs = 20
    n_threads = 4
    W0 = numpy.random.rand(c,d)
    B_values = [8,16,30,60,200,600,3000]
    t3s = []
    for B_size in B_values:
        B = B_size
        print("Batch size: ", B_size)

        t3 = time.time()
        model3_w = sgd_mss_with_momentum_threaded(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs, n_threads)
        t3 = time.time() - t3
        print("\ttime for sgd_mss_with_momentum_threaded:", t3)

        err3 = multinomial_logreg_error(Xs_te, Ys_te, model3_w)
        print("\terror1: ", err3)
        t3s.append(t3)
    t3s = numpy.asarray(t3s)
    t3s_filepath = f'manual_threaded.npy'
    numpy.save(t3s_filepath, t3s)
    plot_time_pt3(t3s, B_values)

def part4(Xs_tr, Ys_tr, Xs_te, Ys_te):
    (d, n) = Xs_tr.shape
    (c, n) = Ys_tr.shape
    alpha = 0.1
    beta = 0.9
    gamma = 0.0001
    num_epochs = 20
    W0 = numpy.random.rand(c,d)
    n_threads = 4
    B_values = [8,16,30,60,200,600,3000]
    # B_values = [8,16]
    t1s = []
    t2s = []
    t3s = []
    for B_size in B_values:
        B = B_size
        print("Batch size: ", B_size)
        t1 = time.time()
        model1_w = sgd_mss_with_momentum_threaded_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs,n_threads)
        t1 = time.time() - t1
        print("\ttime for sgd_mss_with_momentum_noalloc_float32:", t1)

        t2 = time.time()
        # model2_w = sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        t2 = time.time() - t2
        print("\ttime for sgd_mss_with_momentum_threaded_float32:", t2)

        err1 = multinomial_logreg_error(Xs_te, Ys_te, model1_w)
        # err2 = multinomial_logreg_error(Xs_te, Ys_te, model2_w)
        # print("\terror1: ", err1, ".\n\terror2 ", err2)

        t1s.append(t1)
        t2s.append(t2)
    t1s = numpy.asarray(t1s)
    # t2s = numpy.asarray(t2s)
    t1s_filepath = f'no_alloc_float32_1_core_implicit.npy'
    # t2s_filepath = f'no_alloc_float32_1_core_explicit.npy'
    numpy.save(t1s_filepath, t1s)
    # numpy.save(t2s_filepath, t2s)
    # plot_time_pt3(t1s, t2s, B_values)

def plot_all():
    B_values = [8, 16, 30, 60, 200, 600, 3000]
    pyplot.figure(figsize=(12, 8))
    t1s_single_core_filepath = 't1s_1_core.npy'
    t2s_single_core_filepath = 't2s_1_core.npy'
    t1s_multi_core_filepath = 't1s_4_core.npy'
    t2s_multi_core_filepath = 't2s_4_core.npy'
    ts_manual_threading_filepath = 'manual_threaded.npy'
    plot6_data_filepath = 'no_alloc_float32_1_core_no_implicit.npy'
    plot7_data_filepath = 'no_alloc_float32_1_core_explicit.npy'
    plot8_data_filepath = 'no_alloc_float32_1_core_implicit.npy'
    t1s_single_core = numpy.load(t1s_single_core_filepath)
    t2s_single_core = numpy.load(t2s_single_core_filepath)
    t1s_multi_core = numpy.load(t1s_multi_core_filepath)
    t2s_multi_core = numpy.load(t2s_multi_core_filepath)
    ts_manual_threading = numpy.load(ts_manual_threading_filepath)
    p6 = numpy.load(plot6_data_filepath)
    p7 = numpy.load(plot7_data_filepath)
    p8 = numpy.load(plot8_data_filepath)
    t1_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_single_core)
    t1_plot.set_label("w/ alloc w/ 1 core w/o manual threading")
    t2_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_single_core)
    t2_plot.set_label("no alloc w/ 1 core w/o manual threading")
    t3_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t1s_multi_core)
    t3_plot.set_label("w/ alloc w/ 4 cores w/o manual threading")
    t4_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), t2s_multi_core)
    t4_plot.set_label("no alloc w/ 4 cores w/o manual threading")
    t5_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), ts_manual_threading)
    t5_plot.set_label('no alloc w/ 1 core w/ manual threading')
    t6_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), p6)
    t6_plot.set_label('no alloc w/ 1 core w/o manual threading w/ 32-bit float')
    t7_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), p7)
    t7_plot.set_label('no alloc w/ 1 core w/ manual threading w/ 32-bit float')
    t8_plot, = pyplot.plot(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), p8)
    t8_plot.set_label('no alloc w/ 4 core w/o manual threading w/ 32-bit float')
    pyplot.title("Time comparisons of preallocation and threading")
    pyplot.xticks(np.linspace(np.min(B_values), np.max(B_values), len(B_values)), B_values)
    pyplot.xlabel("B_values")
    pyplot.ylabel("Time used (s)")
    pyplot.legend()
    pyplot.savefig("Impact of Preallocation and Threading and Precision")
    pyplot.plot()
    # pyplot.show()
    # pyplot.clf()


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # part12(Xs_tr, Ys_tr, Xs_te, Ys_te)
    # part3(Xs_tr, Ys_tr, Xs_te, Ys_te)
    # part4(Xs_tr, Ys_tr, Xs_te, Ys_te)
    plot_all()

    # TODO add code to produce figures
