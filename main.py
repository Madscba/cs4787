#Test file
import numpy as np
from numpy import random

def softmax(u):
    u_exp = np.exp(u)
    return  u_exp/ sum(u_exp)

def cross_entropy(y_hat,y):
    return -np.sum( y * np.log(y_hat) )

def reg_em_risk(w, x,y_hat,y,gamma):
    activations = np.matmul(w, x)
    y_hat = softmax(activations)
    loss = cross_entropy(y_hat,y)
    loss_norm = loss * 1 / len(y)

    reg_term = gamma/2 * np.linalg.norm(w,ord="fro")



if __name__ == "__main__":
    n = 100 #100
    d = 30 #30
    w = np.arange(d)
    x = np.ones((n, 1)) * random.randint(0, 5, size=d) # (n,d)
    y = random.randint(0, 5, size=n)




