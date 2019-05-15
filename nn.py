import numpy as np
import matplotlib.pyplot as plt
import struct
import os

print("Import all the packages without any error!!")

## load data from mnist

def load_data():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype = np.uint8)
    with open('train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, rows, columns = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype = np.uint8).reshape(num, 784)
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype = np.uint8)
    with open('t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, row, columns = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype = np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels

## visualize data

def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows = 8, ncols = 8, sharex = True, sharey = True)
    ax = ax.flatten()
    for label in range(10):
        print('label = ', label)
        for index in range(64):
             img = img_array[label_array == label][index].reshape(28, 28)
             ax[index].imshow(img, cmap = 'Greys', interpolation = 'nearest')
        plt.show()



## sigmoid function and etc

def sigmoid(x):
    a = np.exp(-x)
    return 1/(1 + a)

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s*(1 - s)

def vis_sigmoid():
    x = np.arange(-10, 10, 0.01)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

def vis_sigmoid_grad():
    x = np.arange(-10, 10, 0.01)
    y = sigmoid_gradient(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    
def adding_bias(X, position):
    if position == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif position == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    return X_new
## Use cross-entropy as the cost function

def cross_entropy_cost(y_enc, output):

    part1 = -y_enc * np.log(output)
    part2 = (1 - y_enc)*np.log(1 - output)
    cost = np.sum(part1 -  part2)
    return cost

## one hot encoding

def one_hot_function(y, n_labels = 10):
    one_hot = np.zeros((n_labels, y.shape[0]))
    for i, value in enumerate(y):
        one_hot[value, i] = 1
    return one_hot


def init_weights(num_input, num_hidden, num_output):
    w1 = np.random.uniform(-1, 1, size = num_hidden*(num_input + 1))
    w1 = w1.reshape(num_hidden, num_input + 1)
    w2 = np.random.uniform(-1, 1, size = num_hidden*(num_hidden + 1))
    w2 = w2.reshape(num_hidden, num_hidden + 1)
    w3 = np.random.uniform(-1, 1, size = num_output*(num_hidden + 1))
    w3 = w3.reshape(num_output, num_hidden + 1)
    return w1, w2, w3

## Foward path

def forward_path(x, w1, w2, w3):
    a1 = adding_bias(x, 'column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)

    a2 = adding_bias(a2, 'row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)

    a3 = adding_bias(a3, 'row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4

## Prediction function

def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = forward_path(x, w1, w2, w3)
    y_predict = np.argmax(a4, axis = 0)
    return y_predict

## Gradient descent

def gradient(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4 - y_enc
    z3 = adding_bias(z3, 'row')
    delta3 = w3.T.dot(delta4) * sigmoid_gradient(z3)
    delta3 = delta3[1:, :]

    z2 = adding_bias(z2, 'row')
    delta2 = w2.T.dot(delta3) * sigmoid_gradient(z2)
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3


def train(X_train, y_train, X_test, y_test):
    X, y = X_train.copy(), y_train.copy()
    y_enc = one_hot_function(y_train)
    epochs = 1000
    batch = 32

    w1, w2, w3 = init_weights(784, 75, 10)
    alpha = 0.01
    eta = 0.01
    discount = 0.001
    delta_w1_previous = np.zeros(w1.shape)
    delta_w2_previous = np.zeros(w2.shape)
    delta_w3_previous = np.zeros(w3.shape)
    total_cost = []
    pred_acc = np.zeros(epochs)

    for i in range(epochs):
        shuffle = np.random.permutation(y.shape[0])
        X, y_enc = X[shuffle], y_enc[:, shuffle]
        eta = eta/(1 + discount*i)

        mini_batch = np.array_split(range(y.shape[0]), batch)
        for iteration in mini_batch:
            a1, z2, a2, z3, a3, z4, a4 = forward_path(X[iteration], w1, w2, w3)
            cost = cross_entropy_cost(y_enc[:, iteration], a4)
            total_cost.append(cost)

            ## Backpropagation

            grad1, grad2, grad3 = gradient(a1, a2, a3, a4, z2, z3, z4, y_enc[:, iteration], w1, w2, w3)
            delta_w1, delta_w2, delta_w3 = eta*grad1, eta*grad2, eta*grad3

            ## Updating weights

            w1 -= delta_w1 + alpha*delta_w1_previous
            w2 -= delta_w2 + alpha*delta_w2_previous
            w3 -= delta_w3 + alpha*delta_w3_previous

            delat_w1_previous, delta_w2_previous, delta_w3_previous = delta_w1, delta_w2, delta_w3

        y_predict = predict(X_test, w1, w2, w3)
        pred_acc[i] = 100*np.sum(y_test == y_predict, axis = 0)/X_test.shape[0]
        print('Epoch :', i)
    return total_cost, pred_acc, y_predict


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_data()
    cost, acc, y_predict = train(X_train, y_train, X_test, y_test)
    value_accuracy = [i for i in range(acc.shape[0])]
    value_cost = [i for i in range(len(cost))]

    print('Prediction accuracy is:', acc[999])
    plt.figure(1)
    plt.plot(value_accuracy, acc)
    plt.title('accuracy')
    plt.figure(2)
    plt.plot(value_cost, cost)
    plt.title('cost function')
    plt.show()


    ## Show prediction result

    test_img = X_test[y_test != y_predict][:25]
    correct_label = y_test[y_test != y_predict][:25]
    pred_label = y_predict[y_test != y_predict][:25]

    fig, ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
    ax = ax.flatten()
    for i in range(25):
        img = test_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_label[i], pred_label[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()






############################# Test ###############################
## Test sigmoid function and its gradient
# vis_sigmoid()
# vis_sigmoid_grad()

## test load function and visualize function
# train_x, train_y, test_x, test_y = load_data()
# print(train_y)
# visualize_data(train_x, train_y)
# print('Done!')
print('Damn Daniel!')
