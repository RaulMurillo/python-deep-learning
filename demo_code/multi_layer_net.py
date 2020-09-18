import numpy as np
from collections import OrderedDict
from demo_code.layers import *
from demo_code.gradient import numerical_gradient

class MultiLayerNet:
    """Fully connected multilayer neural network

    Parameters
    ----------
    input_size : Input size (784 in the case of MNIST)
    hidden_size_list : List of the number of neurons in the hidden layers (e.g. [100, 100, 100])
    output_size : Output size (10 in the case of MNIST)
    activation : 'relu' or 'sigmoid'
    weight_init_std : Specify the standard deviation of the weight (e.g. 0.01)
        Set "Initial value of He" when specifying 'relu' or'he'
        Set "Xavier initial value" when specifying'sigmoid' or 'xavier'
    weight_decay_lambda : Strength of Weight Decay (L2 norm)
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # Weight initializarion
        self.__init_weight(weight_init_std)

        # Layers generation
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """Set the initial value of the weight

        Parameters
        ----------
        weight_init_std : Specify the standard deviation of the weight (e.g. 0.01)
            Set "Initial value of He" when specifying 'relu' or'he'
            Set "Xavier initial value" when specifying'sigmoid' or 'xavier'
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # Recommended initial value when using ReLU
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # Recommended initial value when using sigmoid

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """Find the loss function

        Parameters
        ----------
        x : input data
        t : label data

        Returns
        -------
        The value of the loss function
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Find the gradient (numerical differentiation)

        Parameters
        ----------
        x : input data
        t : label data

        Returns
        -------
        Dictionary variable with gradient of each layer
            grads['W1'], grads['W2'], ...Is the weight of each layer
            grads['b1'], grads['b2'], ...Is the bias of each layer
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """Find the gradient (error back propagation method)

        Parameters
        ----------
        x: input data
        t: label data

        Returns
        -------
        Dictionary variable with gradient of each layer
            grads['W1'], grads['W2'], ...Is the weight of each layer
            grads['b1'], grads['b2'], ...Is the bias of each layer
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # set up
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
