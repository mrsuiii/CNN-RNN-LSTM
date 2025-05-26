import numpy as np
from typing import List
from Value import *


def mse_loss(pred: Value, target: Value) -> Value:
    diff = pred + (-target)  
    return Value(np.mean(diff.data ** 2))

def bce_loss(pred: Value, target: Value) -> Value:
    """
    Binary Cross Entropy Loss:
    loss = - [ target * log(pred) + (1 - target) * log(1 - pred) ], rata-rata elemen.
    Pastikan pred bernilai antara (0,1).
    """
    one = Value(np.ones_like(pred.data))
    loss = -(target * pred.log() + (one + (-target)) * ((one + (-pred)).log()))
    return mean(loss)

def cce_loss(pred: Value, target: Value) -> Value:
    """
    Categorical Cross Entropy Loss:
    Asumsikan pred berisi probabilitas (output softmax) dengan bentuk (batch, num_classes)
    dan target adalah one-hot encoding dengan bentuk yang sama.
    loss = - mean( sum( target * log(pred), axis=1 ) )
    """
    loss = -(target * pred.log())
    loss_sum = sum_axis(loss, axis=1)  # bentuk (batch, 1)
    return mean(loss_sum)

loss_fn_map = {"mse_loss" : mse_loss, "bce_loss" : bce_loss, "cce_loss" : cce_loss}