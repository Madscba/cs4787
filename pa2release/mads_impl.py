from pa1release.main import multinomial_logreg_total_grad
from pa2release.main import multinomial_logreg_grad_i,load_MNIST_dataset,multinomial_logreg_error
from scipy.special import softmax
from tqdm import tqdm
import copy,time, numpy as np,matplotlib

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
    res = [copy.deepcopy(W0)]
    for t in tqdm(range(num_epochs)):
        for i in range(n):
            W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W)
            if ((t*n+i) % monitor_period == monitor_period-1):
                res.append(copy.deepcopy(W))
            if t == 0 and i == 200:
                print("SGD: ", multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W).max(),
                      multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W).min(),
                      multinomial_logreg_grad_i(Xs, Ys, [i], gamma, W).mean())

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
    res = [copy.deepcopy(W0)]
    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            ii = list(range(i * B, i * B + B))
            W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            if t == 0 and i ==0:
                print("SGD_mb: ",multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).max(),multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).min(),multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).mean())
            if ((t*(n/B)+i) % monitor_period == monitor_period-1):
                temp_W = copy.deepcopy(W)
                res.append(copy.deepcopy(temp_W))
    return res

def sgd_minibatch_sequential_scanv2(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    res = [copy.deepcopy(W0)]
    for t in tqdm(range(num_epochs)): #2
        for i in range(int(n/B)): #1000
            ii = list(range(i*B,i*B+B))
            diff = np.zeros_like(W)
            for j in ii:
                diff += - alpha * (1/B) * multinomial_logreg_grad_i(Xs, Ys, [j], gamma, W)
            if t == 0 and i == 0:
                print("SGD_mb_stupid_impl: ", multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).max(),
                      multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).min(),
                      multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W).mean())
            W = W + diff
            if ((t*(n/B)+i) % monitor_period == monitor_period-1):
                temp_W = copy.deepcopy(W)
                res.append(copy.deepcopy(temp_W))


    return res


def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    _, n = Xs.shape
    W = copy.deepcopy(W0)
    res = [copy.deepcopy(W0)]
    for i in tqdm(range(num_iters*10)):
        diff = -alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W)
        W += diff
        # if (i % monitor_freq == 0):
        temp_W = copy.deepcopy(W)
        res.append(copy.deepcopy(temp_W))
        if i == 1:
            print("GD: ", multinomial_logreg_total_grad(Xs, Ys, gamma, W).max(),
                  multinomial_logreg_total_grad(Xs, Ys, gamma, W).min(),
                  multinomial_logreg_total_grad(Xs, Ys, gamma, W).mean())

        #error.append([i,multinomial_logreg_error(Xs,Ys,W_i)])
    # res.append(W_i)
    #matplotlib.pyplot.plot(np.array(error)[:, 0], np.array(error)[:, 1])
    #matplotlib.pyplot.savefig("mygraph.png")
    return res

def plot_function(errors_tr,errors_te):
    errors = [errors_tr,errors_te]
    iterations = errors_tr.shape[1]
    epochs = iterations/10
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
        matplotlib.pyplot.legend([ "SGD_sequential_sampling, final error: {:.3f}".format(errors[i][0,-1]), "Mini_batch_SGD_sequential_sampling, final error: {:.3f}".format(errors[i][1,-1]),"GD, final error: {:.3f}".format(errors[i][2,-1])])
        matplotlib.pyplot.savefig("mads_error_estimate_plot_{}.png".format(["train","test"][i]))
        matplotlib.pyplot.show()

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

    alg2_w = sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_sgd, num_epoch, monitor_period_sgd)

    alg4_w = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_mb_sgd, batch_size, num_epoch, monitor_period_mb_sgd)
    alg4_wv2 = sgd_minibatch_sequential_scanv2(Xs_tr, Ys_tr, gamma, W0, alpha_mb_sgd, batch_size, num_epoch, monitor_period_mb_sgd)

    # algo5_w = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha_mb_sgd*5, num_epoch, monitor_period_mb_sgd)

    errors_tr = np.zeros((3, 10 * num_epoch + 1))
    errors_te = np.zeros((3, 10 * num_epoch + 1))

    alg_weights = [alg2_w, alg4_w,alg4_wv2]
    for i in tqdm(range(3)):
        errors_tr[i, :] = [multinomial_logreg_error(Xs_tr, Ys_tr, w) for w in alg_weights[i]]
        errors_te[i, :] = [multinomial_logreg_error(Xs_te, Ys_te, w) for w in alg_weights[i]]

    plot_function(errors_tr, errors_te)
    pass