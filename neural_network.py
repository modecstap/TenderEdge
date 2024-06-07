import random

import numpy as np
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.INPUT_DIM = input_dim
        self.OUT_DIM = output_dim
        self.H_DIM = hidden_dim

        self.W1 = np.random.rand(self.INPUT_DIM, self.H_DIM)
        self.b1 = np.random.rand(1, self.H_DIM)
        self.W2 = np.random.rand(self.H_DIM, self.OUT_DIM)
        self.b2 = np.random.rand(1, self.OUT_DIM)

        self.W1 = (self.W1 - 0.5) * 2 * np.sqrt(1 / self.INPUT_DIM)
        self.b1 = (self.b1 - 0.5) * 2 * np.sqrt(1 / self.INPUT_DIM)
        self.W2 = (self.W2 - 0.5) * 2 * np.sqrt(1 / self.H_DIM)
        self.b2 = (self.b2 - 0.5) * 2 * np.sqrt(1 / self.H_DIM)

        self.loss_arr_train = []
        self.accuracy_arr_train = []
        self.loss_arr_test = []
        self.accuracy_arr_test = []

    def relu(self, t):
        return np.maximum(t, 0)

    def softmax(self, t):
        out = np.exp(t)
        return out / np.sum(out)

    def softmax_batch(self, t):
        out = np.exp(t)
        return out / np.sum(out, axis=1, keepdims=True)

    def sparse_cross_entropy(self, z, y):
        return -np.log(z[0, y])

    def sparse_cross_entropy_batch(self, z, y):
        return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

    def to_full(self, y, num_classes):
        y_full = np.zeros((1, num_classes))
        y_full[0, y] = 1
        return y_full

    def to_full_batch(self, y, num_classes):
        y_full = np.zeros((len(y), num_classes))
        for j, yj in enumerate(y):
            y_full[j, yj] = 1
        return y_full

    def relu_deriv(self, t):
        return (t >= 0).astype(float)

    def forward(self, x):
        t1 = x @ self.W1 + self.b1
        h1 = self.relu(t1)
        t2 = h1 @ self.W2 + self.b2
        z = self.softmax_batch(t2)
        return z, h1

    def backward(self, x, y, z, h1):
        t1 = x @ self.W1 + self.b1  # Calculate t1
        y_full = self.to_full_batch(y, self.OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)

        dE_dh1 = dE_dt2 @ self.W2.T

        dE_dt1 = dE_dh1 * self.relu_deriv(t1)  # Use t1 here
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)
        return dE_dW1, dE_db1, dE_dW2, dE_db2

    def update(self, dE_dW1, dE_db1, dE_dW2, dE_db2, alpha):
        self.W1 -= alpha * dE_dW1
        self.b1 -= alpha * dE_db1
        self.W2 -= alpha * dE_dW2
        self.b2 -= alpha * dE_db2

    def train(self, train_dataset, test_dataset, alpha, num_epochs, batch_size):
        for ep in tqdm(range(num_epochs)):
            random.shuffle(train_dataset)
            random.shuffle(test_dataset)
            for i in range(len(train_dataset) // batch_size):
                batch_x, batch_y = zip(*train_dataset[i * batch_size: i * batch_size + batch_size])
                x_train = np.concatenate(batch_x, axis=0)
                y_train = np.array(batch_y)

                # Forward pass - training data_train
                z_train, h1_train = self.forward(x_train)

                # Backward pass
                dE_dW1, dE_db1, dE_dW2, dE_db2 = self.backward(x_train, y_train, z_train, h1_train)

                # Update weights
                self.update(dE_dW1, dE_db1, dE_dW2, dE_db2, alpha)

            x_train, y_train = zip(*train_dataset)
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.array(y_train)
            z_test = self.predict(x_train)
            E_train = np.sum(self.sparse_cross_entropy_batch(z_test, y_train))
            accuracy_train = self.calc_accuracy(train_dataset)

            self.loss_arr_train.append(E_train)
            self.accuracy_arr_train.append(accuracy_train)

            x_test, y_test = zip(*test_dataset)
            x_test = np.concatenate(x_test, axis=0)
            y_test = np.array(y_test)
            z_test = self.predict(x_test)
            E_test = np.sum(self.sparse_cross_entropy_batch(z_test, y_test))
            accuracy_test = self.calc_accuracy(test_dataset)

            self.accuracy_arr_test.append(accuracy_test)
            self.loss_arr_test.append(E_test)

    def predict(self, x):
        t1 = x @ self.W1 + self.b1
        h1 = self.relu(t1)
        t2 = h1 @ self.W2 + self.b2
        z = self.softmax_batch(t2)
        return z

    def calc_accuracy(self, dataset):
        correct = 0
        for x, y in dataset:
            z = self.predict(x)
            y_pred = np.argmax(z)
            if y_pred == y:
                correct += 1
        acc = correct / len(dataset)
        return acc

    def set_weigths(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
