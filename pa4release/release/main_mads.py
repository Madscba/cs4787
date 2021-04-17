#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import time
import matplotlib.pyplot as plt
plt.style.use('bmh')
import mnist
import pickle
# matplotlib.use('agg')
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, BatchNormalization

mnist = tf.keras.datasets.mnist


### hyperparameter settings and other constants
batch_size = 128
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings

def KerasModel_init(optim = "SGD",momentum=False,batch_norm=False,alpha=0.1,beta=0.9,addi_layer=0,depth = None):
    """"Mads made this - Just to let you guys know that you can freely adjust the code here! :-) """
    model = Sequential()
    model.add(Flatten())
    if not batch_norm:
        model.add(Dense(units=d1, input_shape=mnist_input_shape, activation="relu"))
        for extra_layer in range(addi_layer):
            model.add(Dense(units=depth, activation="relu"))
        model.add(Dense(units=d2, activation="relu"))
        model.add(Dense(units=num_classes, activation='softmax'))
    else:
        model.add(Dense(units=d1, input_shape=mnist_input_shape, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=d2, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=num_classes, activation='softmax'))
        model.add(BatchNormalization())
    if optim == "SGD":
        if momentum:
            optimizer = keras.optimizers.SGD(learning_rate=alpha, momentum=beta, nesterov=True)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=alpha, momentum=0.0, nesterov=False)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=alpha_adam, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    return model

# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    mnist = tf.keras.datasets.mnist
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()
    Xs_tr = Xs_tr / 255.0
    Xs_te = Xs_te / 255.0
    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 28, 28, 1) # 28 rows, 28 columns, 1 channel
    Xs_te = Xs_te.reshape(Xs_te.shape[0], 28, 28, 1)

    return (Xs_tr, Ys_tr, Xs_te, Ys_te)


# evaluate a trained model on MNIST data, and print the usual output from TF
#
# Xs        examples to evaluate on
# Ys        labels to evaluate on
# model     trained model
#
# returns   tuple of (loss, accuracy)
def evaluate_model(Xs, Ys, model):
    (loss, accuracy) = model.evaluate(Xs, Ys)
    return (loss, accuracy)


# train a fully connected two-hidden-layer neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs,momentum=False,addi_layer=0,depth=None):
    model = KerasModel_init(momentum=momentum,addi_layer=addi_layer,depth=depth)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys,batch_size=B,epochs=epochs,verbose=0,validation_split = validation_split)
    model.summary()
              # validation_data=(X_test, y_test))
    return model, train_history
    # TODO students should implement this

# train a fully connected two-hidden-layer neural network on MNIST data using Adam, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, B, epochs):
    model = KerasModel_init(optim="Adam")
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, verbose=0, validation_split=validation_split)
    model.summary()
    # validation_data=(X_test, y_test))
    return model, train_history

    # TODO students should implement this

# train a fully connected two-hidden-layer neural network with Batch Normalization on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs,addi_layer=0,depth=None):
    model = KerasModel_init(optim="SGD",momentum=True,batch_norm=True,alpha=alpha,beta=beta,addi_layer=addi_layer,depth=depth)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, verbose=0, validation_split=validation_split)
    #model.summary()
    return model, train_history
    # TODO students should implement this


# train a convolutional neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, epochs):
    pass
    # TODO students should implement this

def plot_train_val(train_histories,legends,scores_loss,scores_acc,part):
    for i in range(2):
        for j in range(len(legends)):
            plt.close()
            plt.annotate
            plt.figure(figsize=(10, 8))
            plt.title("{} {}".format(legends[j],["Losses","Accuracies"][i]))
            plt.xlabel("Number of epochs")
            plt.ylabel("{}".format(["Loss", "Accuracy"][i]))

            if i == 0:
                a, b = "loss","val_loss"
                score = scores_loss
            else:
                a, b = "sparse_categorical_accuracy", "val_sparse_categorical_accuracy"
                score = scores_acc
            plt.plot(train_histories[j].history[a])
            plt.plot(train_histories[j].history[b])
            pyplot.xticks(range(0, len(train_histories[j].history[a])))
            plt.hlines(xmin=0,xmax=len(train_histories[j].history[a])-1,  y=score[j],colors='purple',linestyles='dashed')
            plt.legend(['Training {}'.format(["Loss", "Accuracy"][i]),'Validation {}'.format(["Loss", "Accuracy"][i]),"Final {}: {:.3f}".format(["Loss", "Accuracy"][i],score[j])])
            plt.savefig("Part_{}_{}_{}.png".format(part,legends[j],["Losses","Accuracies"][i]))
            plt.show()

        # matplotlib.pyplot.legend([legend[k] + ", final: {:.3f}".format(plot_data[i][k, -1]) for k in range(len(legend))])
        # matplotlib.pyplot.legend([legend[0] + ", final: {:.3f}".format(plot_data[i][0, -1]),
        #                           legend[1] +", final: {:.3f}".format(plot_data[i][1, -1]),
        #                           legend[2] + ", final: {:.3f}".format(plot_data[i][2, -1])])
        # matplotlib.pyplot.savefig(
        #     "Part"+str(part)+"_{}{}.png".format(["TrainingError", "TestError", "TrainingLoss"][i],len(legend)))
        # matplotlib.pyplot.show()



    # plt.plot(train_history.history['loss'], label='Training Loss')
    # plt.plot(train_history.history['val_loss'], label='Validation Loss')
    # plt.title(title)
    # plt.ylabel(ylabel)
    # plt.xlabel('No. epoch')
    # plt.legend(loc="upper left")
    # score = evaluate_model(Xs_te, Ys_te, MLP)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    # plt.hlines(xmin=0,xmax=len(train_history.history['loss']),  y=score[0],colors='purple',
    #            label='vline_multiple - full height')
    # # matplotlib.pyplot.savefig(f"Part_{part}_{title}.png")
    #
    # plt.show()

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # Ys_tr = keras.utils.to_categorical(Ys_tr, num_classes)
    # Ys_te = keras.utils.to_categorical(Ys_te, num_classes)
    MLP, train_history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, batch_size, epochs)


    # TODO students should add code to generate plots here

    raise("REMEMBER TO COMMENT IN THE AGG")
    raise("REMEMBER TO REMOVE MOMENTUM ARGUMENT FROM FIRST SGD")