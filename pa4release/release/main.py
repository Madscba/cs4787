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

def KerasModel_init(d1,d2,optim = "SGD",batch_norm=False,alpha=0.1,beta=0.0,rho1=rho1,rho2=rho2):
    """"Mads made this - Just to let you guys know that you can freely adjust the code here! :-) """
    # addi_layer = 4
    # depth = 128
    model = Sequential()
    model.add(Flatten())
    if not batch_norm:
        model.add(Dense(units=d1, input_shape=mnist_input_shape, activation="relu"))
        # for extra_layer in range(addi_layer): # For part 2, hyperparam optimization
        #     model.add(Dense(units=depth, activation="relu"))
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
        optimizer = keras.optimizers.SGD(learning_rate=alpha, momentum=beta)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=alpha_adam, beta_1=rho1, beta_2=rho2)
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
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    model = KerasModel_init(d1,d2,optim = "SGD",batch_norm=False,alpha=alpha,beta=beta,rho1=0,rho2=0)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys,batch_size=B,epochs=epochs,validation_split = validation_split)
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
    model = KerasModel_init(d1,d2,optim="Adam", batch_norm = False,alpha=alpha,beta=0,rho1=rho1,rho2=rho2)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, validation_split=validation_split)
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
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    model = KerasModel_init(d1,d2,optim="SGD",batch_norm=True,alpha=alpha,beta=beta,rho1 = 0, rho2 = 0)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, validation_split=validation_split)
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
    # TODO students should implement this

    # BUILD MODEL
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))  # dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # dense layer
    opt = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=rho1, beta_2=rho2)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # Compile with SGD optimizer

    # TRAIN MODEL
    start = time.time()
    history = model.fit(Xs, Ys,
                        batch_size=B,
                        epochs=epochs,
                        validation_split=0.1)
    end = time.time()
    print("Time to train: ", end - start)
    return model, history

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

def part_1(Xs, Ys,Xs_te, Ys_te):
    #part 1.1
    MLP_sgd, train_history_sgd = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, 0, batch_size, epochs)
    score_sgd = evaluate_model(Xs_te, Ys_te, MLP_sgd)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    #part 1.2
    MLP_sgd_mom, train_history_sgd_mom = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    score_sgd_mom = evaluate_model(Xs_te, Ys_te, MLP_sgd_mom)
    #
    # #part 1.3
    MLP_adam, train_history_adam = train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, batch_size, epochs)
    score_adam = evaluate_model(Xs_te, Ys_te, MLP_adam)
    #
    # # part 1.4
    MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)


    histories = [train_history_sgd,train_history_sgd_mom,train_history_adam,train_history_bn_sgd]
    legends = ["SGD no momentum","SGD with momentum","Adam","SGD momentum & batch_norm"]
    scores_loss = [score_sgd[0],score_sgd_mom[0],score_adam[0],score_bn_sgd[0]]
    scores_acc = [score_sgd[1],score_sgd_mom[1],score_adam[1],score_bn_sgd[1]]

    plot_train_val(histories, legends,scores_loss,scores_acc,part=1)

    #walltimes
    # num_runs = 5
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs, momentum=False)
    # sum_time1 = time.time() - t1
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs, momentum=True)
    # sum_time2 = time.time() - t1
    #
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, batch_size, epochs)
    # sum_time3 = time.time() - t1
    #
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    # sum_time4 = time.time() - t1

    # print(f"alg1: {sum_time1/num_runs}, alg2: {sum_time2/num_runs}, alg3: {sum_time3/num_runs}, alg4: {sum_time4/num_runs}")

####VERSION FOR PART 2####
def KerasModel_init2(optim = "SGD",momentum=False,batch_norm=False,alpha=0.1,beta=0.9,addi_layer=0,depth = None):
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
def train_fully_connected_sgd2(Xs, Ys, d1, d2, alpha, beta=beta, B=batch_size, epochs=epochs,momentum=False,addi_layer=0,depth=None):
    model = KerasModel_init2(momentum=momentum,addi_layer=addi_layer,depth=depth)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys,batch_size=B,epochs=epochs,verbose=0,validation_split = validation_split)
    model.summary()
              # validation_data=(X_test, y_test))
    return model, train_history

def train_fully_connected_adam2(Xs, Ys, d1, d2, alpha, rho1, rho2, B, epochs):
    model = KerasModel_init2(optim="Adam")
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, verbose=0, validation_split=validation_split)
    model.summary()
    # validation_data=(X_test, y_test))
    return model, train_history

def train_fully_connected_bn_sgd2(Xs, Ys, d1, d2, alpha, beta, B, epochs,addi_layer=0,depth=None):
    model = KerasModel_init2(optim="SGD",momentum=True,batch_norm=True,alpha=alpha,beta=beta,addi_layer=addi_layer,depth=depth)
    validation_split = 0.1
    train_history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, verbose=0, validation_split=validation_split)
    #model.summary()
    return model, train_history


def part_2(Xs, Ys,Xs_te, Ys_te,alpha=0.1,beta=0.9):
    # Part 2.1
    best_alpha = 100
    best_acc = 0

    alpha_values = [1.0,0.3,0.1,0.03,0.01,0.003,0.001]
    val_acc = []
    val_loss = []
    for alpha in alpha_values:
        MLP_sgd_mom, train_history_sgd_mom = train_fully_connected_sgd2(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,momentum=True)
        val_acc.append(train_history_sgd_mom.history["val_sparse_categorical_accuracy"][-1])
        val_loss.append(train_history_sgd_mom.history["val_loss"][-1])
        if val_acc[-1] > best_acc:
            best_alpha = alpha
            best_acc = val_acc[-1]
    print("val_acc: ",val_acc)
    print("val_loss: ",val_loss)
    print("Best alpha and acc: {} {}".format(best_alpha,best_acc))
    # Part 2.2
    best_vaL_acc = 0

    additional_layers = [1,2,4]
    additional_layers_depth = [128,256,512]
    beta_values = [0.5,0.9,0.99,0.999]
    for additional_layer in additional_layers:
        for additional_layer_depth in additional_layers_depth:
            for beta in beta_values:
                MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_sgd2(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,addi_layer=additional_layer,depth=additional_layer_depth)
                score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)
                if train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1] > best_vaL_acc:
                    best_hyperparams = {"extra_layers":additional_layer,"depth":additional_layer_depth,"beta":beta}
                    best_vaL_acc = train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1]
                    best_val_loss = train_history_bn_sgd.history["val_loss"][-1]
                    best_test_loss = score_bn_sgd[0]
                    best_test_acc = score_bn_sgd[1]

    print("Best hyperparams: ",best_hyperparams)
    print("val acc: {}, val loss: {}".format(best_vaL_acc,best_val_loss))
    print("test acc: {}, test loss: {}".format(best_test_acc, best_test_loss))


    # part 2.3 random search
    best_vaL_acc = 0
    combinations_of_hyperparams =  []
    for i in range(3):
        additional_layer = int(np.random.uniform(low=1,high=5,size=1)[0])
        for j in range(3):
            additional_layer_depth = int(np.random.normal(loc=256,scale=128,size=1)[0])
            for k in range(4):
                beta = np.random.normal(loc=.5,scale=0.2,size=1)[0]
                MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_sgd2(Xs, Ys, d1, d2, alpha, beta, batch_size,
                                                                             epochs, addi_layer=additional_layer,
                                                                             depth=additional_layer_depth)
                score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)
                combinations_of_hyperparams.append((additional_layer,additional_layer_depth,beta))
                if train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1] > best_vaL_acc:
                    best_hyperparams = {"extra_layers": additional_layer, "depth": additional_layer_depth, "beta": beta}
                    best_vaL_acc = train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1]
                    best_val_loss = train_history_bn_sgd.history["val_loss"][-1]
                    best_test_loss = score_bn_sgd[0]
                    best_test_acc = score_bn_sgd[1]

    print("Best hyperparams: ", best_hyperparams)
    print("val acc: {}, val loss: {}".format(best_vaL_acc, best_val_loss))
    print("test acc: {}, test loss: {}".format(best_test_acc, best_test_loss))

    print("Tested combinations:",combinations_of_hyperparams)


def plot_metric(tr, val, te, title, metric, epochs):
    pyplot.figure(figsize=(12, 8))
    pyplot.plot(tr, label="Training " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.plot(val, label="Validation " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.plot([te] * epochs, label="Test " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.title(title + metric)
    pyplot.xticks(range(0, epochs))
    pyplot.xlabel("Epochs")
    pyplot.ylabel(metric)
    pyplot.legend()
    pyplot.savefig(title+" "+metric)
    pyplot.clf()

def train_plot(Xs_te, Ys_te, model, history, title, epochs):
    tr_loss = history.history['loss']
    tr_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    te_loss_acc = evaluate_model(Xs_te, Ys_te, model)
    te_loss, te_acc = te_loss_acc
    plot_metric(tr_loss, val_loss, te_loss, title, 'Loss', epochs)
    plot_metric(tr_acc, val_acc, te_acc, title, 'Accuracy', epochs)

def part_3(Xs_tr, Ys_tr, Xs_te, Ys_te):
    model, history = train_CNN_sgd(Xs_tr, Ys_tr, alpha_adam, rho1, rho2, batch_size, epochs)
    train_plot(Xs_te, Ys_te, model, history, "Convolutional Neural Network", 10)


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()


    # TODO students should add code to generate plots here
    # part_1(Xs_tr, Ys_tr,Xs_te, Ys_te)
    # part_2(Xs_tr, Ys_tr,Xs_te, Ys_te)
    # part_3(Xs_tr, Ys_tr,Xs_te, Ys_te)
