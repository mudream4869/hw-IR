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

learning_rate = .0001/qc
alpha = 0.000001
iter_time = 100000

while True:
    w = np.random.rand(137)*2 - 1
    v = mypack.cal_ndcg(w, v_list)
    print("Cal V", v)
    if v > 0.28 : break

print("Choose w with ndcg in valid data =", v)

X = []
Y = []

for qid, doc_list in (t_list+v_list):
    for rel, arr, docid in doc_list:
        X.append(arr)
        Y.append([rel])

X = np.array(X)
Xt = np.transpose(X)
Y = np.array(Y)

XtX = np.dot(Xt, X) + np.diag([0] + [1]*136)*0.0001
XtY = np.dot(Xt, Y)

YY = np.transpose(Y).dot(Y)

print(XtX.shape, XtY.shape)

# (y - wx)^2 = y^2 - 2yx^Tw + wxx^Tw

result = open("task2-result.csv", "w")
result.write("Iteration,Loss,Train Data NDCG\n")

for it in range(iter_time):
    dw = learning_rate*((XtX.dot(w.reshape(137, 1)) - XtY).reshape(137, ))
    w -= dw
       
    if (it+1)%100 == 0 :
        loss = YY - 2*w.dot(XtY) + w.dot(XtX).dot(w)
        print("Loss :", loss[0][0])
        print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
        result.write("%d,%f,%f\n" % (it+1, loss[0][0], mypack.cal_ndcg(w, t_list+v_list)) )

np.savetxt("task2-iter.model", w)
result.close()

print("Train NDCG:", mypack.cal_ndcg(w, t_list))
print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
