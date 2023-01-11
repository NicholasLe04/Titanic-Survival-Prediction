import numpy as np

class NeuralNetwork():

    def __init__(self, nn_architecture, seed=99):
        np.random.seed(seed)

        self.nn_arch = nn_architecture
        self.params_values = {}

        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            self.params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            self.params_values['B' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
            

    def single_layer_forward_propagation(self, A_prev, W_curr, B_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + B_curr
        
        if activation == "relu":
            activation_func = self.relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        elif activation == "leaky relu":
            activation_func = self.leaky_relu
        elif activation == "tanh":
            activation_func = self.tanh
        else:
            raise Exception('Non-supported activation function')
            
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X):
        memory = {}
        A_curr = X
        
        for idx, layer in enumerate(self.nn_arch):
            layer_idx = idx + 1
            A_prev = A_curr
            
            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            B_curr = self.params_values["B" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, B_curr, activ_function_curr)
            
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        return A_curr, memory


    def single_layer_backward_propagation(self, dA_curr, W_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]
        
        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        elif activation == "leaky relu":
            backward_activation_func = self.leaky_relu_backward
        elif activation == "tanh":
            backward_activation_func = self.tanh_backward
        else:
            raise Exception('Non-supported activation function')
        
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        dB_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, dB_curr

    def full_backward_propagation(self, Y_hat, Y, memory):
        gradient_values = {}
        Y = Y.reshape(Y_hat.shape)
    
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_arch))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            
            dA_curr = dA_prev
            
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = self.params_values["W" + str(layer_idx_curr)]
            
            dA_prev, dW_curr, dB_curr = self.single_layer_backward_propagation(dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)
            
            gradient_values["dW" + str(layer_idx_curr)] = dW_curr
            gradient_values["dB" + str(layer_idx_curr)] = dB_curr
        
        return gradient_values


    ## Adjusts weights and biases by moving in the opposite direction of gradient
    def update(self, grads_values, learning_rate):
        for layer_idx in range(1, len(self.nn_arch)):
            self.params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
            self.params_values["B" + str(layer_idx)] -= learning_rate * grads_values["dB" + str(layer_idx)]

        return self.params_values


    def train(self, X, Y, epochs, learning_rate):
        cost_history = []
        accuracy_history = []
        
        for i in range(epochs):
            Y_hat, cashe = self.full_forward_propagation(X)
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = self.full_backward_propagation(Y_hat, Y, cashe)
            self.params_values = self.update(grads_values, learning_rate)
            
        return self.params_values, cost_history, accuracy_history


    # ACTIVATION FUNCTIONS

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0,Z)

    def leaky_relu(self, Z):
        return np.maximum(0.1*Z, Z)

    def tanh (self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        
    # ACTIVATION FUNCTION DERIVATIVES

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0;
        return dZ;

    def leaky_relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = dA[Z <= 0] * 0.01;
        return dZ;

    def tanh_backward(self, dA, Z):
        t = self.tanh(Z)
        return dA * (1 - (t * t))


    ## LOSS CALCULATION

    def get_cost_value(self, Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def convert_prob_into_class(self, probs):
            probs_ = np.copy(probs)
            probs_[probs_ > 0.5] = 1
            probs_[probs_ <= 0.5] = 0
            return probs_


    def predict (self, test_input):
        return self.full_forward_propagation(test_input)[0][0][0]


## RUNNING ##




    