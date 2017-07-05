import math
import numpy as np

import random 

def sigmoid(x):
    if x < -200:
        return 0
    return 1/(1+math.exp(-x))

def dsigmoid(x):
    if x < -200:
        return 0
    return math.exp(-x)/(1+math.exp(-x))**2

def task1(_items):
    w, (qid, doc_list), p_list = _items
    loss = 0
    d_loss = np.zeros(137)
    rg = range(len(doc_list))
    wdot = [np.dot(x[1], w) for x in doc_list]
    fx = [sigmoid(wdot[i]) for i in rg]
    dfx = [dsigmoid(wdot[i]) * x[1] for i, x in enumerate(doc_list)]

    for i, j in p_list:
        eji = math.exp(fx[j] - fx[i])
        d_loss += eji/(1 + eji)*(dfx[j]-dfx[i]) 
        loss += math.log(1 + eji)

    return d_loss, loss
