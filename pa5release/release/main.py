#!/usr/bin/env python3
import os
import numpy
import numpy as np
from numpy import random
import scipy
from scipy import special
import matplotlib
import mnist
import pickle
import math
from matplotlib import pyplot
import matplotlib.animation as animation

from tqdm import tqdm
from scipy.special import softmax

import tensorflow as tf

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants
d = 1
gamma = 10
sigma2_noise = 0.001
random_x = np.random.rand
gd_nruns = 20
gd_alpha = 0.01
gd_niters = 20
n_warmup = 3
num_iters = 20
kappa = 2
### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
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
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,0:50000]
        Ys_tr = Ys_tr[:,0:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset

# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + tf.math.erf(u/numpy.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return tf.math.exp(-u**2/2.0)/numpy.math.sqrt(2.0*numpy.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in TensorFlow)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
#
def rbf_kernel_matrix(Xs,Zs,gamma):
    d1, m = Xs.shape
    d2, n = Zs.shape
    # assert (d1 == d2), "Dimensions of input vectors must match!"

    D = -2 * tf.matmul(Xs, Zs, transpose_a=True) #Was set to true
    s1 = tf.reduce_sum(Xs ** 2, axis=0)
    s2 = tf.reduce_sum(Zs ** 2, axis=0)
    #print((tf.transpose(s1)).dtype,tf.ones(n,dtype=tf.float64).dtype)
    S = tf.tensordot(tf.transpose(s1), tf.ones(n,dtype=tf.float64),axes=0)
    R = tf.tensordot(tf.ones(m,dtype=tf.float64), tf.transpose(s2),axes=0)
    G = tf.matmul(Xs, Zs, transpose_a=True)
    D = S + R - 2 * G
    # D = tf.maximum(D, 0)
    D = tf.math.exp(-gamma * tf.abs(D))
    return D
# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in TensorFlow)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # first, do any work that can be shared among predictions
    # TODO students should implement this
    Sigma_inv = tf.linalg.inv( rbf_kernel_matrix(Xs, Xs, gamma) + tf.eye(Xs.shape[1],dtype=tf.float64) * sigma2_noise )
    general_term = tf.matmul(Sigma_inv, Ys)
    # next, define a nested function to return
    def prediction_mean_and_variance(Xtest):
        # TODO students should implement this
        # construct mean and variance
        k_star_T = tf.transpose(rbf_kernel_matrix(Xs,Xtest,gamma))
        mean = tf.matmul(k_star_T,general_term)
        variance = rbf_kernel_matrix(Xtest,Xtest,gamma) + sigma2_noise - tf.matmul(tf.matmul(k_star_T ,Sigma_inv), np.transpose(k_star_T))
        return (mean, variance)
    #finally, return the nested function
    return prediction_mean_and_variance


# compute the probability of improvement (PI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    Z = (Ybest-mean) / stdev
    return - gaussian_cdf(Z)


# compute the expected improvement (EI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    Z = (Ybest-mean) / stdev
    return - ( gaussian_pmf(Z) + Z * gaussian_cdf(Z) )*stdev


# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        # TODO students should implement this
        return mean - kappa*stdev
    return A_lcb


# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a tensorflow variable and returns an expression
# x0            initial value to assign to variable x
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    x = tf.Variable(x0)
    for it in range(num_iters):
        with tf.GradientTape() as tape:
            f = objective(x)
            (g, ) = tape.gradient(f, [x])
        x.assign(x - alpha * g)
    return (float(f), x)

# run Bayesian optimization to minimize an objective
#
# objective     objective function
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    x_best = np.NaN
    y_best = np.infty
    ys = []
    xs = []
    for x in range(n_warmup):
        x_i = tf.convert_to_tensor(random_x(d),dtype=tf.float64)
        y_i = objective(x_i)
        xs.append(x_i)
        ys.append(y_i)
        print(x_i.shape)
        if y_i <= y_best:
            x_best = x_i
            y_best = y_i
            print("Warmup: x_i: {}, y_i: {}".format(float(x_best),float(y_best)))

    def inner_opt_obj(x):
            # tmpXs = xs.copy()
            # tmpYs = ys.copy()
            # mean_variance_func = gp_prediction(Xs, Ys, gamma, sigma2_noise)
            mean, Sigma = mean_variance_func(x)
            return acquisition(mean, Sigma, y_best)


    # Xs = tf.transpose(tf.convert_to_tensor(xs,dtype=tf.float64) )
    # if len(np.shape(Xs)) < 2:
    #     Xs = tf.reshape(Xs, [1,Xs.shape[0]])
    # if len(np.shape(ys)) > 2:
    #     Ys = tf.convert_to_tensor(tf.squeeze(ys,-1),dtype=tf.float64)
    # else:
    #     Ys = tf.convert_to_tensor(ys,dtype=tf.float64)
    for i in range(n_warmup,num_iters):
        Xs = tf.transpose(tf.convert_to_tensor(xs,dtype=tf.float64) )
        if len(np.shape(Xs)) < 2:
            Xs = tf.reshape(Xs, [1,Xs.shape[0]])
        if len(np.shape(ys)) > 2:
            Ys = tf.convert_to_tensor(tf.squeeze(ys,-1),dtype=tf.float64)
        else:
            Ys = tf.convert_to_tensor(ys,dtype=tf.float64)
        
        mean_variance_func = gp_prediction(Xs, Ys, gamma, sigma2_noise)
        x_i = np.NaN
        acq_best = np.infty
        for j in range(gd_nruns):
            x_init = tf.convert_to_tensor(random_x(d).reshape(-1,1),dtype=tf.float64)
            obj_min, x_min = gradient_descent(inner_opt_obj, x_init, gd_alpha, gd_niters)
            if obj_min <= acq_best:
                x_i = x_min
                acq_best = obj_min
        y_i = objective(x_i)
        if y_i <= y_best:
            x_best = x_i
            y_best = y_i
            print("Actual bayes: x_i: {}, y_i: {}".format(float(x_best),float(y_best)))
        xs.append(tf.squeeze ( tf.convert_to_tensor(x_i,dtype=tf.float64),1))

        ys.append(tf.squeeze(y_i),1)
    return y_best,x_best,ys,xs
    # TODO students should implement this


# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    pass
    return (numpy.cos(8.0*x) - 0.3 + (x-0.5)**2)


# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
#
# objective     objective function
# acq           acquisition function
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       list of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file 
def animate_predictions(objective, acq, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    acq_eval = []
    acq_Xnext = []
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1)]
        ybest = Ysi.min()
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        pred_acqs = []
        XE = tf.Variable(numpy.zeros((1,)))
        for x_eval in xs_eval:
            XE.assign(numpy.array([x_eval]))
            (pred_mean, pred_variance) = gp_pred(XE)
            pred_means.append(float(pred_mean))
            pred_variances.append(float(pred_variance))
            pred_acqs.append(float(acq(ybest, pred_mean, tf.math.sqrt(pred_variance))))
        mean_eval.append(numpy.array(pred_means))
        variance_eval.append(numpy.array(pred_variances))
        acq_eval.append(numpy.array(pred_acqs))
        if it + 1 != len(Ys):
            XE.assign(Xs[:, it+1])
            (pred_mean, pred_variance) = gp_pred(XE)
            acq_Xnext.append(float(acq(ybest, pred_mean, tf.math.sqrt(pred_variance))))

    fig = pyplot.figure()
    fig.tight_layout()
    ax = fig.gca()
    ax2 = ax.twinx()

    def animate(i):
        ax.clear()
        ax2.clear()
        ax.set_xlabel("parameter")
        ax.set_ylabel("objective")
        ax2.set_ylabel("acquisiton fxn")
        ax.set_title("Bayes Opt After %d Steps" % (i+1))
        l1 = ax.fill_between(xs_eval, mean_eval[i] + 2.0*numpy.sqrt(variance_eval[i]), mean_eval[i] - 2.0*numpy.sqrt(variance_eval[i]), color="#eaf1f7")
        l2, = ax.plot(xs_eval, objective(xs_eval))
        l3, = ax.plot(xs_eval, mean_eval[i], color="r")
        l4 = ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        l5, = ax2.plot(xs_eval, acq_eval[i], color="g", ls=":")
        ax.legend([l2, l3, l5], ["objective", "mean", "acquisition"], loc="upper right")
        if i + 1 == len(Ys):
            return l1, l2, l3, l4, l5
        else:
            l6 = ax2.scatter([Xs[0,i+1]], [acq_Xnext[i]], color="g")
            return l1, l2, l3, l4, l5, l6


    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), interval=400, repeat_delay=1000)

    ani.save(filename)


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 2
    return numpy.dot(softmax(numpy.dot(W, Xs[:,ii]), axis=0) - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 1
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error


# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 3
    (d, n) = Xs.shape
    return -numpy.sum(numpy.log(softmax(numpy.dot(W, Xs), axis=0)) * Ys) / n + (gamma / 2) * (numpy.linalg.norm(W, "fro")**2)


# SGD + Momentum: run stochastic gradient descent with minibatching, sequential sampling order, and momentum (SAME AS PROGRAMMING ASSIGNMENT 3)
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
    pass
    # TODO students should use their implementation from programming assignment 3


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    pass
    # TODO students should implement this
def random_xs(d):
    random_uni_initializer = tf.random_uniform_initializer(minval=0, maxval=1, seed=None)
    return tf.Variable(random_uni_initializer(shape=[d, 1], dtype=tf.float32))

def part_2_12(acq_ind):
    
    # Set objective
    objective = test_objective
    
    # Acquisition functions and names
    acq_funcs = [pi_acquisition, ei_acquisition, lcb_acquisition(kappa)]
    acq_func_str = ["Probability of improvement aquisition (pi)",
                    "Expected improvement aquisition (ei)",
                    "Lower confidence bound (lcb, kappa=", kappa, ")"]
    
    # Track previous values
    all_y_best = []
    all_x_best = []
    all_Ys = []
    all_Xs = []
    
    # Run Bayesian opt for each acquisition function
    for i in range(len(acq_funcs)):
        print("Running Bayesian optimization with acquisition function ", acq_func_str[i], ".")
        y_best, x_best, Ys, Xs = bayes_opt(objective, d, gamma, sigma2_noise, acq_funcs[i],
                                            random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        print("\t Best parameter value: ", float(x_best))
        print("\t Best objective value: ", float(y_best))
        all_y_best.append(y_best)
        all_x_best.append(x_best)
        all_Ys.append(Ys)
        all_Xs.append(Xs)
            
    # Choosing acquisition to animate
    acq = acq_funcs[acq_ind]
    Ys = all_Ys[acq_ind]
    Xs = all_Xs[acq_ind]
    xs_eval = Xs[0]
    filename = "PrA5_p2_video.mp4"
    #animate_predictions(objective, acq, gamma, sigma2_noise, Ys, Xs, xs_eval, filename)
    
    return all_y_best, all_x_best

def part_2_3(acq_ind, gamma_vals, og_y_best, og_x_best, k = -1):
    
    # Set objective
    objective = test_objective
    
    # Acquisition functions and names
    acq_funcs = [pi_acquisition, ei_acquisition, lcb_acquisition(k)]
    acq_func_str = ["Probability of improvement aquisition (pi)",
                    "Expected improvement aquisition (ei)",
                    "Lower confidence bound (lcb, kappa=", k, ")"]
    
    # Set acquisition function
    acq = acq_funcs[acq_ind]
    
    print("Original Bayesian optimization with acquisition function ", acq_func_str[acq_ind], ", gamma=", gamma, ".")
    print("\t Best parameter value: ", float(og_x_best))
    print("\t Best objective value: ", float(og_y_best))
    
    # track best param & obj
    all_x_best = []
    all_y_best = []
    
    # for every gamma, run Baysian optimization
    for g in gamma_vals:
        print("Running Bayesian optimization with acquisition function ", acq_func_str[acq_ind], ", gamma=", g, ".")
        y_best, x_best, Ys, Xs = bayes_opt(objective, d, g, sigma2_noise, acq,
                                            random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        print("\t Best parameter value: ", float(x_best))
        print("\t Best objective value: ", float(y_best))
        all_x_best.append(x_best)
        all_y_best.append(y_best)
        
    print("\n All best parameter values: ", [float(x) for x in  all_x_best])
    print("All best objective values: ", [float(y) for y in  all_y_best])

def part_2_4(kappa_vals, og_y, og_x):
    
    # Set objective
    objective = test_objective
    
    print("Original Bayesian optimization with acquisition function lcb, kappa=",kappa,".")
    print("\t Best parameter value: ", float(og_x))
    print("\t Best objective value: ", float(og_y))
    
    # track best param & obj
    all_x_best = []
    all_y_best = []
    
    # for every kappa, run Baysian optimization
    for k in kappa_vals:
        acq = lcb_acquisition(k)
        print("Running Bayesian optimization with acquisition function lcb, kappa=", k, ".")
        y_best, x_best, Ys, Xs = bayes_opt(objective, d, kappa, sigma2_noise, acq,
                                            random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        print("\t Best parameter value: ", float(x_best))
        print("\t Best objective value: ", float(y_best))
        all_x_best.append(x_best)
        all_y_best.append(y_best)
        
    print("\n All best parameter values: ", [float(x) for x in all_x_best])
    print("All best objective values: ", [float(y) for y in all_y_best])

def part_2():
    acq_ind_to_video = 0
    acq_ind_to_ex = 0
    gamma_vals = [1, 20, 100]
    kappa_vals = [1, 5, 10]
    
    # Parts 1 and 2
    all_y_best, all_x_best = part_2_12(acq_ind_to_video)
    
    # # Part 3
    # part_2_3(acq_ind_to_ex, gamma_vals, all_y_best[acq_ind_to_ex], all_x_best[acq_ind_to_ex])
    
    # # Part 4
    # part_2_4(kappa_vals, all_y_best[2], all_x_best[2])

if __name__ == "__main__":
    d,n,m = 1,5,6
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)
    # Xs = tf.Variable(initializer(shape=[d, m], dtype=tf.float32))
    # Zs = tf.Variable(initializer(shape=[d, n], dtype=tf.float32))
    # Ys = tf.Variable(initializer(shape=[m, 1], dtype=tf.float32))



    # gp_prediction(Xs,Ys,gamma,2)
    # RBFkernel = rbf_kernel_matrix(Xs, Zs, gamma)
    # print(np.array(RBFkernel))
    pass
    # gamma = 10
    # sigma2_noise = 0.001
    # gd_nruns, gd_alpha, gd_niters = 100, 0.05, 100
    # n_warmup, num_iters = 3,20
    # kappa = 2
    #Part 2.1
    # #(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    # best_x_pi = bayes_opt(test_objective, d, gamma, sigma2_noise, pi_acquisition, random_xs, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    # best_x_ei = bayes_opt(test_objective, d, gamma, sigma2_noise, ei_acquisition, random_xs, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    # best_x_lcb = bayes_opt(test_objective, d, gamma, sigma2_noise, lcb_acquisition(kappa), random_xs, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    # print("pi: {}, ei: {}, lcb: {}".format(float(test_objective(best_x_pi)),float(test_objective(best_x_ei)),float(test_objective(best_x_lcb))))
    a = 2
    # RBFkernel = rbf_kernel_matrix(Xs, Xs, gamma)
    part_2()

    # TODO students should implement plotting functions here

