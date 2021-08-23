# !/usr/bin/env python
# coding: utf-8


from tensorflow.keras.datasets import mnist
import random
import sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def multiple_experiments(x_train, y_train, x_test, y_test, layers_dims, learning_rate, num_iterations):
    '''
    Runs multiple experiments and examine connection between batch size
    and test accuracy/run-time

    Input: x_train, y_train, x_test, y_test, layers_dims, learning_rate, num_iterations

    Output: overall summary, plots
    '''
    global show_summary
    batch_sizes = [100, 200, 300, 400, 500, 1000]
    predictions, lapsed_times = list(), list()
    show_summary = False

    for batch_size in batch_sizes:
        start = time.time()
        parameters, costs = l_layer_model(x_train, y_train, layers_dims, learning_rate, num_iterations, batch_size)
        end = time.time()
        lapsed_times.append((end - start))
        predictions.append(predict(x_test, y_test, parameters))

    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, predictions, 'bo')
    plt.title("Accuracy Vs. Batch Size")
    plt.ylabel("Accuracy (%)")
    plt.xlabel('Batch Sample Size')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, lapsed_times, 'bo')
    plt.title("Run-Time Vs. Batch Size")
    plt.ylabel("Run-Time (sec)")
    plt.xlabel('Batch Sample Size')
    plt.show()


def after_training_info(val_accuracy, epochs):
    # Prints a summary of the model after training
    print(' ')
    print('Overall Performance:')
    print('------------------------------------------')
    print('Number of epochs is ' + str(epochs))
    print('Validation accuracy: ' + str(round(val_accuracy, 2)))


def after_testing_info(train_accuracy, test_accuracy):
    # Prints a summary when testing
    print('Training accuracy: ' + str(round(train_accuracy, 2)))
    print('Test accuracy: ' + str(round(test_accuracy, 2)))


def plot_costs(training_costs, validation_costs):
    # summarize history for loss
    plt.figure(figsize=(18, 8))
    plt.plot(training_costs)
    plt.plot(validation_costs)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Step index (every 100 iterations)')
    plt.legend(['train cost', 'validation cost'], loc='upper left')
    plt.show()


def load_data():
    '''
    Description: Loading the data

    Input: None

    Output: X_train, y_train, X_test, y_test
    '''
    # loads mnist data, Converts y data to one hot form and normalize x
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # flattening input to 3D -> 2D (28x28 -> 1x784)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).T
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).T

    # normalize input
    x_train = x_train.astype(np.float64) / 255.
    x_test = x_test.astype(np.float64) / 255.

    # convert y to one-hot-encoding
    y_train_temp = np.zeros((y_train.max() + 1, y_train.size))
    y_train_temp[y_train, np.arange(y_train.size)] = 1
    y_train = y_train_temp
    y_test_temp = np.zeros((y_test.max() + 1, y_test.size))
    y_test_temp[y_test, np.arange(y_test.size)] = 1
    y_test = y_test_temp

    return (x_train, y_train), (x_test, y_test)


def train_test_split_implementation(X, Y, test_proportion):
    '''
    Description: Shuffles the data and splits to test and train

    Input: X, Y and the tes proportion

    Output: X_train, y_train, X_test, y_test
    '''
    # shape of both X,Y = (number of records, number of features)

    # shuffle randmoly both x and y (IN UNISION)
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    # split
    ratio = int(X.shape[0] / test_proportion)  # should be int
    X_train = X[ratio:, :]
    X_test = X[:ratio, :]
    Y_train = Y[ratio:, :]
    Y_test = Y[:ratio, :]
    return X_train.T, Y_train.T, X_test.T, Y_test.T


def initialize_parameters(layer_dims):
    '''
    input: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output softmax)

    output: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).

    '''
    parameters = dict()
    np.random.seed(42)
    for idx, layer in enumerate(layer_dims):
        if idx == 0:
            continue
        else:
            w_mat = np.random.randn(layer, layer_dims[idx - 1]) * np.sqrt(2 / layer_dims[idx - 1])
            parameters['W' + str(idx)] = w_mat
            b_mat = np.zeros((layer, 1))
            parameters['b' + str(idx)] = b_mat

    return parameters


def linear_forward(A, W, b):
    '''
    Description: Implement the linear part of a layer's forward propagation.

    input:
    A – the activations of the previous layer
    W – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    B – the bias vector of the current layer (of shape [size of current layer, 1])

    Output:
    Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
    linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    '''
    linear_cache = {'A': A, 'W': W, 'b': b}
    Z = np.dot(W, A) + b
    return Z, linear_cache


def softmax(Z):
    '''
    Input:
    Z – the linear component of the activation function

    Output:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation

    note:use
    Softmax can be thought of as a sigmoid for multi-class problems.
    The formula for softmax for each node in the output layer is as follows:
    Softmax〖(z)〗_i=(exp⁡(z_i))/(∑_j▒〖exp⁡(z_j)〗)
    '''
    c = np.max(Z)
    activation_cache = Z
    A = np.exp(Z - c) / sum(np.exp(Z - c))

    return (A, activation_cache)


def relu(Z):
    '''
    Input:
    Z – the linear component of the activation function

    Output:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    '''
    A = np.copy(Z)
    A[A < 0] = 0
    return A, Z


def linear_activation_forward(A_prev, W, B, activation):
    '''
    Description:
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Input:
    A_prev – activations of the previous layer
    W – the weights matrix of the current layer
    B – the bias vector of the current layer
    Activation – the activation function to be used (a string, either “softmax” or “relu”)

    Output:
    A – the activations of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    '''
    Z, linear_cache = linear_forward(A_prev, W, B)

    if activation == "softmax":
        A, activation_cache = softmax(Z)
    if activation == "relu":
        A, activation_cache = relu(Z)

    cache = dict()
    cache['activation_cache'] = activation_cache
    cache['linear_cache'] = linear_cache

    return A, cache


def l_model_forward(X, parameters, use_batchnorm):
    '''
    Description:
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

    Input:
    X – the data, numpy array of shape (input size, number of examples)
    parameters – the initialized W and b parameters of each layer
    use_batchnorm - a boolean flag used to determine whether to apply batchnorm after the activation (note that this option needs to be set to “false” in Section 3 and “true” in Section 4).

    Output:
    AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    '''

    A_current = X
    caches = list()
    idx = 1
    num_layers = int(len(parameters) / 2)

    while idx != num_layers:
        A_current, cache_current = linear_activation_forward(A_current, parameters['W' + str(idx)],
                                                             parameters['b' + str(idx)], 'relu')
        caches.append(cache_current)
        idx += 1
        if use_batchnorm:
            A_current = apply_batchnorm(A_current)
        if use_dropout and in_training:
            A_current = apply_dropout(A_current, 0.1)

    AL, cache_last = linear_activation_forward(A_current, parameters['W' + str(idx)], parameters['b' + str(idx)],
                                               'softmax')
    caches.append(cache_last)
    return AL, caches


def compute_cost(AL, Y):
    '''
    Description:
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss. The formula is as follows :
    cost=-1/m*∑_1^m▒∑_1^C▒〖y_i  log⁡〖(y ̂)〗 〗, where y_i is one for the true class (“ground-truth”) and y ̂ is the softmax-adjusted prediction (this link provides a good overview).

    Input:
    AL – probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    Y – the labels vector (i.e. the ground truth)

    Output:
    cost – the cross-entropy cost
    '''
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)
    epsilon = 1e-9
    ce = -(np.sum(Y * np.log(AL + epsilon)) / m)
    ce = np.squeeze(ce)

    return ce


def apply_batchnorm(A):
    '''
    Description:
    performs batchnorm on the received activation values of a given layer.

    Input:
    A - the activation values of a given layer

    output:
    NA - the normalized activation values, based on the formula learned in class
    '''
    mean = np.mean(A)
    var = np.var(A)
    norm = (A - mean) / np.sqrt(var + np.finfo(float).eps)
    return norm


def linear_backward(dZ, cache):
    '''
    Description:
    Implements the linear part of the backward propagation process for a single layer

    Input:
    dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    '''
    global masks, layer_index
    A_prev = cache['A']
    W = cache['W']
    B = cache['b']
    m = A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

    if layer_index >= 0:
        mask = masks[layer_index]
        layer_index -= 1
        dA_prev = dA_prev * mask

    else:  # end of backward (first layer) - initialize for next batch
        masks = []

    return (dA_prev, dW, db)


def linear_activation_backward(dA, cache, activation='relu'):
    '''
    Description:
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.

    Input:
    dA – post activation gradient of the current layer
    cache – contains both the linear cache and the activations cache

    Output:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b

    '''
    linear_cache = cache['linear_cache']  # Aprev,W,B
    activation_cache = cache['activation_cache']  # dictionary in the form of: {Z,Y}

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = softmax_backward(dA, activation_cache)  # Softmax for last layer

    return linear_backward(dZ, linear_cache)


def softmax_backward(dA, activation_cache):
    '''
    Description:
    Implements backward propagation for a softmax unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    '''
    # According to the formula: dZ = p - y
    Z = activation_cache['cache']
    y = activation_cache["Y"]
    p, cache = softmax(Z)

    dZ = p - y

    return dZ


def relu_backward(dA, activation_cache):
    '''
    Description:
    Implements backward propagation for a ReLU unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    '''
    # By the formula: dZ = dcost_dA * dA_dZ
    Z = activation_cache
    arr = np.array(dA, copy=True)
    arr[Z <= 0] = 0
    dZ = arr

    return dZ


def l_model_backward(AL, Y, caches):
    '''
    Description:
    Implement the backward propagation process for the entire network.

    Some comments:
    the backpropagation for the softmax function should be done only once as only the output layers uses it and the RELU should be done iteratively over all the remaining layers of the network.

    Input:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache

    Output:
    Grads - a dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
    '''
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # We will transfer Y to the softmax_backward calculation
    caches[-1]['activation_cache'] = {"cache": caches[-1]['activation_cache'], "Y": Y}

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cache = caches[L - 1]
    # dA is currently dA-prev
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, cache,
                                                                                                      "softmax")
    for l in reversed(range(L - 1)):
        cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    '''
    Description:
    Updates parameters using gradient descent

    Input:
    parameters – a python dictionary containing the DNN architecture’s parameters
    grads – a python dictionary containing the gradients (generated by L_model_backward)
    learning_rate – the learning rate used to update the parameters (the “alpha”)

    Output:
    parameters – the updated values of the parameters object provided as input
    '''
    L = round(len(parameters) / 2)
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


def l_layer_model(X, Y, layer_dims, learning_rate, num_iterations, batch_size):
    '''initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters'''
    global in_training
    costs = []
    costs_val = []
    stop_learning = False
    iter_count = 0
    epsilon = 0.001
    prev_cost_val = math.inf
    epochs = 0

    parameters = initialize_parameters(layer_dims)
    x_train, y_train, x_val, y_val = train_test_split_implementation(X.T, Y.T, 5)  # % 20 val

    train_batches = create_batches(x_train, y_train, batch_size)
    val_batches = create_batches(x_val, y_val, batch_size)


    while (stop_learning==False):
        epochs += 1
        val_batch_index = 0

        for train_batch_index in range(len(train_batches)):
            val_batch_index += 1
            if (val_batch_index == len(val_batches)):
                val_batch_index = 0
            if (iter_count == num_iterations + 1):
                stop_learning = True
                write_costs_to_file(costs, costs_val)
                break
            # train
            X_batch_train, Y_batch_train = train_batches[train_batch_index]
            in_training = True
            AL_train, caches_train = l_model_forward(X_batch_train, parameters, use_batchnorm)
            grads = l_model_backward(AL_train, Y_batch_train, caches_train)
            parameters = update_parameters(parameters, grads, learning_rate)
            in_training = False

            # validation
            X_batch_val, Y_batch_val = val_batches[val_batch_index]
            AL_val, caches_val = l_model_forward(X_batch_val, parameters, use_batchnorm)

            # Examine performance  on validation set and the print costs
            if iter_count % 100 == 0:
                cost_train = compute_cost(AL_train, Y_batch_train)
                costs.append(cost_train)
                cost_val = compute_cost(AL_val, Y_batch_val)
                costs_val.append(cost_val)
                delta_cost = prev_cost_val - cost_val
                if (delta_cost < epsilon and delta_cost > 0):
                    stop_learning = True
                    write_costs_to_file(costs, costs_val)
                    break
                prev_cost_val = cost_val
                val_accuracy = predict(x_val, y_val, parameters)
                print('Iteration: {}, cost = {}, val cost = {}, val_accuracy = {}'.format(iter_count, cost_train,
                                                                                          cost_val, val_accuracy))
            iter_count += 1

    if show_summary == True:
        # plot costs
        plot_costs(costs, costs_val)

        # print model's summary
        after_training_info(val_accuracy, epochs)

    return parameters, costs


def predict(X, Y, parameters):
    '''
    Description:
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.

    Input:
    X – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters

    Output:
    accuracy – the accuracy measure of the neural net on the provided data (i.e. the percentage of the samples for which the correct label receives the highest confidence score).
    Use the softmax function to normalize te output values.
    '''
    # For each vector - extract highest value of softmax as a prediction
    m = Y.shape[1]
    AL, caches = l_model_forward(X, parameters, use_batchnorm)
    y_hat = np.argmax(AL, axis=0)
    Y = np.argmax(Y, axis=0)

    # range is between 0 to 1
    true_predicitions = np.sum(Y == y_hat)
    accuracy = true_predicitions / m

    return accuracy


def apply_dropout(A, drop_probability):
    '''
    Description: Perform dropout to the neurons of a particular layer

    Input: A, drop_out probability

    Output: layer output A with nullified values (silence a neuron by multiplying A values by zero)
    '''
    global masks, layer_index
    A_probs = np.random.rand(A.shape[0], A.shape[1])
    A_probs[A_probs < drop_probability] = 0.0
    A_probs[A_probs >= drop_probability] = 1.0
    A = A * A_probs
    # corects the value of A upwards to correct for the missing nodes
    keep_prob = 1 - drop_probability
    corrected_A = A / keep_prob

    layer_index += 1
    mask = A_probs
    masks.append(mask)

    return corrected_A


def write_costs_to_file(train_cost, val_cost):
    '''
    Description: This function writes the oveall costs (train, validation)
    into a text file.

    Input: train costs, validation costs

    Output: text (.txt) file, with the relevant costs
    '''
    f = open("neural network costs.txt", "w")
    for index, cost in enumerate(train_cost):
        f.write('Iteration: {}, train cost = {}, validation cost ={}\n'.format(((index) * 100), train_cost[index],
                                                                               val_cost[index]))
    f.close()


def create_batches(X, Y, batch_size):
    '''
    Description: This function divides the data into batches.
    It returns an array consisted of tuples - (x_batch, y_batch)

    Input: Data (X,Y) and the desired batch_size

    Output: The data, consisted of batches
    '''
    m = X.shape[1]
    batches = []
    for i in range(0, m, batch_size):
        X_batch = X[:, i:i + batch_size]
        Y_batch = Y[:, i:i + batch_size]
        batches.append((X_batch, Y_batch))

    # When no equal division
    if (m % batch_size != 0):
        X_batch = X[:, i:m]
        Y_batch = Y[:, i:m]
        batches.append((X_batch, Y_batch))

    return batches


def train_net(x, y, layers_dims, learning_rate, iterations, batch_size):
    '''
    Description: Trains the entire network

    Input: Data (X,Y) and the desired batch_size

    Output: The data, consisted of batches
    '''
    #  Measures the training time (in minutes) of the entire network
    start = time.time()
    params, costs = l_layer_model(x, y, layers_dims, learning_rate, iterations,
                                  batch_size)
    end = time.time()
    # Converts to minutes
    training_time = round(((end - start) / 60), 2)
    print('Training took {} minutes'.format(training_time))
    return params, costs, training_time


def calculate_accuracy(x, y, x_test, y_test, params):
    # Calculates accuracy for both train and test sets
    train_accuracy = predict(x, y, params)
    test_accuracy = predict(x_test, y_test, params)

    return train_accuracy, test_accuracy


# main
if __name__ == '__main__':
    # General hyperparameters
    learning_rate = 0.009
    batch_size = 200
    iterations = 50000
    layers_dims = [784, 20, 7, 5, 10]
    use_batchnorm = False  # hardcoded - Change to True in order to perform batchnorm
    show_summary = True

    # Drop out hyperparameters
    use_dropout = False  # hardcoded - Change to True in order to perform dropout
    masks = []
    layer_index = -1
    in_training = True

    # Read data
    (x_train, y_train), (x_test, y_test) = load_data()

    # train the net
    params, costs, training_time = train_net(x_train, y_train, layers_dims, learning_rate,
                                             iterations, batch_size)
    # Evaluate
    train_accuracy, test_accuracy = calculate_accuracy(x_train, y_train, x_test, y_test, params)

    # Display results
    after_testing_info(train_accuracy, test_accuracy)

    # Batch size VS run-time/accuracy experiments
    #multiple_experiments(x_train, y_train, x_test, y_test, layers_dims, learning_rate, iterations)


