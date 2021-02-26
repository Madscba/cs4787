from scipy.special import softmax
from tqdm import tqdm
from matplotlib import pyplot
import os
import numpy as np
import scipy
import matplotlib
import mnist
import pickle
import random,time
matplotlib.use('agg')


mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# additional imports you may find useful for this assignment

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
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    softmax_input = np.matmul(W,x)
    y_hat = softmax(softmax_input)
    l2_reg = (gamma/2) * np.sum(np.dot(W.T,W))
    loss = - np.dot(y.T,np.log(y_hat)) + l2_reg
    return loss + l2_reg
# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters


def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    softmax_input = np.matmul(W,x)
    un_reg_grad = np.outer((softmax(softmax_input)-y),x)
    l2_reg_grad = gamma * np.sum(W)
    return un_reg_grad + l2_reg_grad
# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i


def test_gradient():
    # TODO students should implement this in Part 1
    (X, Y, _, _) = load_MNIST_dataset()
    d, n = X.shape
    c, _ = Y.shape
    W = np.random.rand(c,d)
    gamma,eta = 0.0001, .000005

    gradient_diff = 0
    gradient_mag = 0
    for i in range(10):
        random_idices = random.sample(list(range(n)),1)
        x = X[:,random_idices]
        y = Y[:,random_idices]
        f_x = multinomial_logreg_loss_i(x,y,gamma,W)
        v = (np.zeros_like(W)+1) *eta
        f_x_eta = multinomial_logreg_loss_i(x,y,gamma,W+v)
        grad_apprx = ( f_x_eta - f_x ) / eta
        real_gradient = multinomial_logreg_grad_i(x,y,gamma,W)
        aprrox_grad_sum = np.sum( (v.T.dot(real_gradient)))*grad_apprx

        #v.dot(real_gradient) * grad_apprx #approx real_gradient
        print("real_gradient_sum: {} \t Grad_approx {}".format(aprrox_grad_sum,grad_apprx))
        gradient_diff += abs(real_gradient-grad_apprx)
        gradient_mag += np.linalg.norm(real_gradient)

# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels


def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    incorrect_count = 0
    Xs = Xs.T
    Ys = Ys.T
    for x, y in zip (Xs, Ys):
        softmax_input = np.matmul(W,x)
        pred = np.argmax(softmax(softmax_input))

        incorrect_count += 1 if y[pred] == 0 else 0
    return incorrect_count /  np.shape(Xs)[0]
# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters


def multinomial_logreg_total_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d, n) = Xs.shape

    softmax_input = np.matmul(W,Xs)
    Ys_hat = softmax(softmax_input,axis=0)

    total_grad = np.matmul( (Ys_hat-Ys), Xs.T)
    total_grad / n
    return  total_grad
    # y_hat = np.sum( softmax(softmax_input,axis=0),axis=0)
    #
    # l2_reg = (gamma/2) * np.sum(np.dot(W.T,W))
    # loss = - np.dot(Ys,np.log(y_hat)) + l2_reg
    # return loss

    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_grad_i(Xs[:, i], Ys[:, i], gamma, W)
    # return acc / n

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss


def multinomial_logreg_total_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients

    (d, n) = Xs.shape
    softmax_input = np.matmul(W,Xs)
    Ys_hat = softmax(softmax_input,axis=0)

    total_loss = - np.sum( Ys * np.log(Ys_hat)) / n
    return total_loss
    #
    # res = multinomial_logreg_loss_i(Xs,Ys,gamma, W)
    # np.sum(res) / n
    #
    # acc = 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_loss_i(Xs[:, i], Ys[:, i], gamma, W)
    #
    #
    # est1 = acc / n
    # if np.sumP(est1-total_loss) < 0.001:
    #     print("BAD")
    # return total_loss


# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    res = []
    error = []
    W_i = W0
    for i in tqdm(range(num_iters)):
        if (i % monitor_freq == 0):
            res.append(W_i)
        diff = -alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W_i)
        W_i += diff
        error.append([i,multinomial_logreg_error(Xs,Ys,W_i)])
    res.append(W_i)
    matplotlib.pyplot.plot(np.array(error)[:, 0], np.array(error)[:, 1])
    matplotlib.pyplot.savefig("mygraph.png")
    return res


# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the estimated model error when sampling with replacement
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    (d, n) = Xs.shape
    iterations = np.shape(W)[2]
    exact_errors,est_errs_100,est_errs_1000 = [],[],[]

    t_exact = time.time()
    for w_iter_no in tqdm(range(iterations)):
        W_i = W[:,:,w_iter_no]
        softmax_input = np.matmul(W_i, Xs)
        pred = np.argmax(softmax(softmax_input, axis=0), axis=0)
        lookup_vec = np.array(list(enumerate(pred)))
        correct_predictions = Ys[lookup_vec[..., 1], lookup_vec[..., 0]]
        exact_error = (n - np.sum(correct_predictions)) / n
        exact_errors.append(exact_error)
    t_exact = time.time() - t_exact

    t_100 = time.time()
    for w_iter_no in tqdm(range(iterations)):
        W_i = W[:,:,w_iter_no]
        rand_idx_100 = random.sample(list(range(n)), nsamples[0])
        X_subsample_100 = Xs[:, rand_idx_100]
        Y_subsample_100 = Ys[:, rand_idx_100]
        softmax_input = np.matmul(W_i, X_subsample_100)
        pred = np.argmax(softmax(softmax_input, axis=0), axis=0)
        lookup_vec = np.array(list(enumerate(pred)))
        correct_predictions = Y_subsample_100[lookup_vec[..., 1], lookup_vec[..., 0]]
        est_error = (nsamples[0] - np.sum(correct_predictions)) / nsamples[0]
        est_errs_100.append(est_error)
    t_100 = time.time()-t_100

    t_1000 = time.time()
    for w_iter_no in tqdm(range(iterations)):
        W_i = W[:, :, w_iter_no]
        rand_idx_1000 = random.sample(list(range(n)), nsamples[1])
        X_subsample_1000 = Xs[:, rand_idx_1000]
        Y_subsample_1000 = Ys[:, rand_idx_1000]
        softmax_input = np.matmul(W_i, X_subsample_1000)
        pred = np.argmax(softmax(softmax_input, axis=0), axis=0)
        lookup_vec = np.array(list(enumerate(pred)))
        correct_predictions = Y_subsample_1000[lookup_vec[..., 1], lookup_vec[..., 0]]
        est_error = (nsamples[1] - np.sum(correct_predictions)) / nsamples[1]
        est_errs_1000.append(est_error)
    t_1000 = time.time()-t_1000

    print("Times were exact: {}, 1000: {}, and 100 {}".format(t_exact,t_1000,t_100))

    matplotlib.pyplot.figure(figsize=(15,10))
    matplotlib.pyplot.plot(range(101), exact_errors)
    matplotlib.pyplot.plot(range(101), est_errs_100)
    matplotlib.pyplot.plot(range(101), est_errs_1000)
    matplotlib.pyplot.legend(["Exact error","Error 100_sub","Error 1000_sub"])
    matplotlib.pyplot.title("Exact vs estimate errors with subsampling")
    matplotlib.pyplot.xlabel("Iteration (model version)")
    matplotlib.pyplot.ylabel("Error")
    matplotlib.pyplot.savefig("error_estimate_plot.png")
    return est_error


if __name__ == "__main__":
    test_gradient()
    # np.random.seed(42)
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # print("Xs_tr.shape:", Xs_tr.shape)
    # print("Ys_tr.shape:", Ys_tr.shape)
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    W0 = np.random.rand(c,d)
    a = 2
    gamma=0.0001
    alpha=1.0
    num_iters=10
    monitor_freq=10
    #W_iter = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_iters, monitor_freq)
    W_iter = np.repeat(np.random.rand(c,d),101).reshape(c,d,101)
    test = estimate_multinomial_logreg_error(Xs_tr,Ys_tr,W_iter,[100,1000])

