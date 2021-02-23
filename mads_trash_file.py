import numpy as np
from numpy import random
import random as ran,matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def compute_cost(X, y, theta):
#     m = len(y)
#     h = sigmoid(X @ theta)
#     epsilon = 1e-5
#     cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
#     return cost


def empirical_risk(w, x, y, lambda_):
    em_risk, reg_term = 0, w[0] * w[0]
    for n in range(len(y)):
        temp_risk = w[0] * x[n, 0]
        for d in range(1,len(w)):
            temp_risk += w[d] * x[n, d]
        em_risk += np.log(1 + np.exp(-y[n] * temp_risk))
    em_risk = em_risk * 1 / len(y)
    for d in range(1,len(w)):
        reg_term += w[d] * w[d]
    return em_risk + lambda_ * reg_term

def empirical_risk2(w,x,y,lambda_):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    inner_exp = np.matmul(x,w)
    inner_exp = -y*inner_exp
    em_risk = 1/len(y) * np.sum( np.log( 1+ np.exp(inner_exp)) )
    reg_term = np.dot(w.T,w)
    return em_risk + lambda_ * reg_term

def compute_gradient1(w,x,y,lambda_):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    nominator = -y*x # (n,1) x (n,d) = (n,d)
    demoninator =  (1 + np.exp( y * np.matmul(x,w) ) ).reshape(-1,1) #(n,1)
    return np.sum(nominator / demoninator, axis=0).reshape(-1,1) * 1/ len(y) + 2 * lambda_ * w


    #return np.matmul(x.T, (-y)) * np.exp(1 + (np.matmul(np.matmul(- y,w.T), x.T3)))
def compute_gradient(w,x,y,lambda_):
    g = np.zeros((len(w),1))
    m = 1/ len(y)
    for j in range(len(w)):
        f_gradient_k = 0
        for k in range(len(y)):
            inner_exp = 0
            for i in range(len(w)):
                inner_exp += w[i] * x[k, i]
            f_gradient_k += (-y[k] * x[k,j] * m) / ( 1 + np.exp(y[k]* inner_exp) )
        g[j] = f_gradient_k + 2*lambda_*w[j]
    return g

def stochastic_gradient_descent(w,x,y,lambda_,alpha):
    x_i,y_i = r(x,y)
    g = compute_gradient1(w,x_i,y_i,lambda_)
    new_w = w.reshape(-1,1) - alpha * g
    return new_w

def gradient_descent(w,x,y,lambda_,alpha):
    g = compute_gradient1(w,x,y,lambda_)
    new_w = w.reshape(-1,1) - alpha * g
    return new_w
def r(x,y):
    x_i,y_i = ran.choice(list(zip(x,y)))
    return np.array([x_i]).reshape(1,-1),np.array( [y_i]).reshape(-1,1)

if __name__ == "__main__":
    N = 100
    n = 100 #100
    d = 30 #30
    w = np.arange(d)
    w1 = w
    x = np.ones((n, 1)) * np.arange(d)  # (n,d)
    y = random.randint(-1, 2, size=n)
    y = np.array([d if d == 1 else -1 for d in y])
    lambda_,alpha= 0.05,0.05
    loss = []
    for i in range(N):
        loss.append(empirical_risk2(w,x,y,lambda_))
        for j in range(round(len(y) / 20)):
            w = stochastic_gradient_descent(w,x,y,lambda_,alpha)
    plt.plot(np.arange(N),np.squeeze(loss))

    loss_g = []
    for i in range(N):
        loss_g.append(empirical_risk2(w1,x,y,lambda_))
        w1 = gradient_descent(w1,x,y,lambda_,alpha)
    plt.plot(np.arange(N),np.squeeze(loss_g))
    plt.legend(['SGD','GD'])
    plt.show()

    # print("w: ", np.shape(w), " x: ", np.shape(x), " y: ", np.shape(y))
    # print("risks:", empirical_risk(w, x, y, 2), empirical_risk2(w, x, y, 2), " cost:")
    # delta_x = np.ones((n, 1)) * np.arange(d) * 0.05
    # print("change:", empirical_risk(w, x + +delta_x, y, 2), empirical_risk2(w, x + +delta_x, y, 2), " cost:")
    #
    # print("grad1", compute_gradient(w, x, y, 2), " grad2 ", compute_gradient1(w, x, y, 2))