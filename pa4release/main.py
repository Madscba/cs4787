#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import tensorflow as tf

import time

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
    # TODO students should implement this
    
    # BUILD MODEL
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(d1, activation='relu')) # First dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(d2, activation='relu')) # Second dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Third dense layer
    opt = tf.keras.optimizers.SGD(learning_rate = alpha, momentum = beta)
    model.compile(optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']) # Compile with SGD optimizer
    
    # TRAIN MODEL
    start = time.time()
    history = model.fit(Xs, Ys,
                        batch_size = B,
                        epochs = epochs,
                        validation_split = 0.1)
    end = time.time()
    print("Time to train: ", end - start)
    
    return model, history


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
    # TODO students should implement this
    
    # BUILD MODEL
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(d1, activation='relu')) # First dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(d2, activation='relu')) # Second dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Third dense layer
    opt = tf.keras.optimizers.Adam(learning_rate = alpha, beta_1 = rho1, beta_2 = rho2)
    model.compile(optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']) # Compile with SGD optimizer
    
    # TRAIN MODEL
    start = time.time()
    history = model.fit(Xs, Ys,
                        batch_size = B,
                        epochs = epochs,
                        validation_split = 0.1)
    end = time.time()
    print("Time to train: ", end - start)
    
    return model, history


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
    # TODO students should implement this
    
    # BUILD MODEL
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(d1, activation='relu')) # First dense layer
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(d2, activation='relu')) # Second dense layer
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Third dense layer
    model.add(tf.keras.layers.BatchNormalization())
    opt = tf.keras.optimizers.SGD(learning_rate = alpha, momentum = beta)
    model.compile(optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']) # Compile with SGD optimizer
    
    # TRAIN MODEL
    start = time.time()
    history = model.fit(Xs, Ys,
                        batch_size = B,
                        epochs = epochs,
                        validation_split = 0.1)
    end = time.time()
    print("Time to train: ", end - start)
    
    return model, history


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
    model.add(tf.keras.layers.Conv2D(32, (5,5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (5,5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu')) # dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # dense layer
    opt = tf.keras.optimizers.Adam(learning_rate = alpha, beta_1 = rho1, beta_2 = rho2)
    model.compile(optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']) # Compile with SGD optimizer
    
    # TRAIN MODEL
    start = time.time()
    history = model.fit(Xs, Ys,
                        batch_size = B,
                        epochs = epochs,
                        validation_split = 0.1)
    end = time.time()
    print("Time to train: ", end - start)
    
    return model, history


def train_plot(Xs_te, Ys_te, model, history, title, epochs):
    tr_loss = history.history['loss']
    tr_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    te_loss_acc = evaluate_model(Xs_te, Ys_te, model)
    te_loss, te_acc = te_loss_acc
    plot_metric(tr_loss, val_loss, te_loss, title, 'Loss', epochs)
    plot_metric(tr_acc, val_acc, te_acc, title, 'Accuracy', epochs)

def plot_metric(tr, val, te, title, metric, epochs):
    pyplot.figure(figsize=(12,8))
    pyplot.plot(tr, label= "Training " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.plot(val, label= "Validation " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.plot([te]*epochs, label= "Test " + metric + ", final = " + "{:.3f}".format(tr[-1]))
    pyplot.title(title + metric)    
    pyplot.xticks(range(0,epochs))
    pyplot.xlabel("Epochs")
    pyplot.ylabel(metric)
    pyplot.legend()
    # pyplot.savefig(title+" "+metric)
    pyplot.clf()


def part_1(Xs_tr, Ys_tr, Xs_te, Ys_te):
    model_sgd1, history_sgd1 = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, 0.0, batch_size, epochs)
    model_sgd2, history_sgd2 = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, batch_size, epochs)
    model_adam, history_adam = train_fully_connected_adam(Xs_tr, Ys_tr, d1, d2, alpha_adam, rho1, rho2, batch_size, epochs)
    model_bn, history_bn = train_fully_connected_bn_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, batch_size, epochs)
    train_plot(Xs_te, Ys_te, model_sgd1, history_sgd1, "Fully Connected SGD", epochs)
    train_plot(Xs_te, Ys_te, model_sgd2, history_sgd2, "Fully Connected SGD w Momentum", epochs)
    train_plot(Xs_te, Ys_te, model_adam, history_adam, "Fully Connected SGD w Adam", epochs)
    train_plot(Xs_te, Ys_te, model_bn, history_bn, "Fully Connected SGD w Batch Normalization", epochs)


def part_2(Xs_tr, Ys_tr, Xs_te, Ys_te):
    model, history = train_CNN_sgd(Xs_tr, Ys_tr, alpha_adam, rho1, rho2, batch_size, epochs)
    train_plot(Xs_te, Ys_te, model, history, "Convolutional Neural Network", 10)


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO students should add code to generate plots here
    part_1(Xs_tr, Ys_tr, Xs_te, Ys_te)
    part_2(Xs_tr, Ys_tr, Xs_te, Ys_te)

