import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(100, 3)

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class LossCategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class ActivationSoftmaxCCE:
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        samples = len(inputs)
        y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, y_true):
        samples = len(self.output)
        labels = self.output.shape[1]

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # derivative
        self.dinputs = (self.output - y_true) / samples


class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()

dense2 = LayerDense(64, 3)
activation2 = ActivationSoftmax()

# activation2 = ActivationSoftmax()
# loss_function = LossCategoricalCrossentropy()


loss_activation = ActivationSoftmaxCCE()

optimizer = OptimizerSGD()

def train(X, y, epochs=1000, batch_size=32, print_every=100):
    samples = X.shape[0]
    loss_activation = ActivationSoftmaxCCE()

    dense1 = LayerDense(2, 64)
    activation1 = ActivationReLU()
    dense2 = LayerDense(64, 3)
    optimizer = OptimizerSGD(learning_rate=0.1)

    for epoch in range(1, epochs + 1):
        for i in range(0, samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)

            loss = loss_activation.forward(dense2.output, y_batch)
            predictions = np.argmax(loss_activation.output, axis=1)
            acc = np.mean(predictions == y_batch)

            loss_activation.backward(y_batch)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            optimizer.update_params(dense1)
            optimizer.update_params(dense2)

        if epoch % print_every == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

train(X, y)
