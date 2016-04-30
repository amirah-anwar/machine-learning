# This creates the network
import numpy as np

class Network(object):
    def __init__(self, layers):
        self.bias = [np.random.randn(x, 1) for x in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers, layers[1:])]

        print(self.bias)
        print(self.weights)

    def sigmoid(self, activations):
        return 1 / (1 + np.exp(-1 * activations))

    def sigmoid_prime(self, activations):
        return np.array(self.sigmoid(activations)) * np.array((1 - self.sigmoid(activations)))

    def forward(self, inputs):
        inputs = inputs.reshape(2, 1) # How many inputs ?
        z = [inputs]
        for weights, bias in zip(self.weights, self.bias):
            z_layer = np.dot(weights, inputs) + bias
            z.append(z_layer)
            inputs = self.sigmoid(z_layer)
        return z, inputs

    def evaluate(self, result, target):
        if target == 1:
            target = np.array([[0], [1]])
        else:
            target = np.array([[1], [0]])
        return result - target

    def backprop(self, input, result, target, activations):
        new_weights = [np.zeros(elem.shape) for elem in self.weights]
        new_biases = [np.zeros(elem.shape) for elem in self.bias]

        # calculate the delta for all layers starting from last layer
        delta = []
        sig_prime = self.sigmoid_prime(activations[-1])
        error = self.evaluate(result, target)
        delta_last = np.array(sig_prime) * np.array(error)
        for index in range(len(new_weights) - 1, -1, -1):
            new_weights[index] = np.transpose(self.sigmoid(activations[index]) * np.transpose(delta_last))
            new_biases[index] = delta_last
            if index != 0:
                delta_last = np.array(np.transpose(self.weights[index]) * np.matrix(delta_last)) * np.array(self.sigmoid_prime(activations[index]))

        return new_weights, new_biases

    def sgd(self, weight, bias, eta):
        for i in range(len(weight)):
            self.weights[i] = self.weights[i] - eta * np.matrix(weight[i])
            self.bias[i] = self.bias[i] - eta * np.array(bias[i])

    def predict(self, test_data, test_label):
        counter = 0
        for input, target in zip(test_data, test_label):
            _, result = self.forward(input)
            if result[0] > result[1]:
                if target == 0:
                    counter += 1
            elif result[1] > result[0]:
                if target == 1:
                    counter += 1
        print('The number of correct %d out of %d' %(counter, len(test_label)))

    def train(self, train, test=None, eta = 0.5, epoch = 1):
        # For now we will do gradient descent for each sample.
        # TODO add mini batch
        train_data = train[0]
        train_label = train[1]
        if test:
            self.predict(test[0], test[1])
        count = 0
        while count < epoch:
            indices = np.random.permutation(train_data.shape[0])
            train_data = train_data[indices]
            train_label = train_label[indices]
            for inputs, target in zip(train_data, train_label):
                activations, result = self.forward(inputs)
                weight, bias = self.backprop(inputs, result, target, activations)
                self.sgd(weight, bias, eta)

            if test:
                print('Epoch %d' % (count))
                self.predict(test[0], test[1])
            count += 1

