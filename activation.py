from Value import Value
import numpy as np

def exp(x: Value):
    t = np.exp(x.data)
    out = Value(t)
    out._prev = {x}
    
    def _backward():
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        x.grad += t * out.grad

    out._backward = _backward
    out._op = 'exp'
    return out

def linear(x:Value):
    return x

def tanh(x: Value):
    exp_x = exp(x)
    exp_neg_x = 1 / exp_x
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def relu(x: Value):
    return x * (x.data > 0)

def leaky_relu(x: Value, alpha=0.01):
    out = x * (x.data > 0) + Value(alpha) * x * (x.data <= 0)
    return out

def swish(x: Value):
    return x * sigmoid(x)

def sigmoid(x: Value):
    return 1 / (1 + exp(-x))

def softmax(x: Value):
    max_x = Value(np.max(x.data, axis=1, keepdims=True))
    exp_x = exp(x - max_x)
    sum_exp = Value(np.sum(exp_x.data, axis=1, keepdims=True))
    out = exp_x / sum_exp
    
    out._prev = {x}
    
    def _backward():
        # Compute softmax Jacobian-vector product
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        
        # Gradient computation for softmax
        # s * (δ_ij - s_j)
        # out.data is the softmax probabilities
        # out.grad is the incoming gradient
        grad = out.data * (out.grad - np.sum(out.data * out.grad, axis=1, keepdims=True))
        
        x.grad += grad

    out._backward = _backward
    out._op = 'softmax'
    return out

activations_map = {"linear" : linear, "tanh" : tanh, "relu" : relu, "leaky_relu" : leaky_relu, "sigmoid" : sigmoid, "swish" : swish, "softmax" : softmax}