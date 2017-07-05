import numpy as np
import math
import random
import itertools

from multiprocessing import Pool

import iters
import mypack

import pickle
query_map = pickle.load(open("/tmp2/b02902029/train.test.normalize.pickle", "rb"))
print("Read normalized file ok.")

t_list, v_list = mypack.split_tv(query_map)

qc = len(t_list)

print("Train/Valid :", len(t_list), len(v_list))

X = []
Y = []

for qid, doc_list in (t_list):
    nor = len(doc_list)**0.5
    for rel, arr, docid in doc_list:
        X.append(arr/nor)
        Y.append([1.5**rel/nor])

X = np.array(X)
Xt = np.transpose(X)
Y = np.array(Y)

XtX = np.dot(Xt, X) + np.diag([1]*137)*0.0001
XtY = np.dot(Xt, Y)

from numpy.linalg import inv
w = np.dot(np.dot(inv(XtX), Xt), Y).reshape((137, ))

np.savetxt("task3-cf.model", w)

print("Train NDCG:", mypack.cal_ndcg(w, t_list))
print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
