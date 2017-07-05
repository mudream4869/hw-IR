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
    if v > 0.25 : break

print("Choose w with ndcg in valid data =", v)

I = np.diag([1]*137)
P, Q = I, np.array([[0]]*137)
# w' = Pw + Q

for qid, doc_list in (t_list):
    A, B = [], []
    for rel, arr, docid in doc_list:
        A.append(arr)
        B.append([1.5**rel])
    A, B = map(np.array, [A, B])
    At = A.T
    AtA = Xt.dot(X) + I*0.0001
    AtB = Xt.dot(Y)
    BB = B.T.dot(B)

    # 
    (Pw + Q)(I-A)

result = open("task3-result.csv", "w")
result.write("Iteration,Loss,Train Data NDCG\n")

# w'   = w - Aw + B
#      = (I-A)w + B
# w'-K = (I-A)(w-K)
# w'   = (I-A)w + AK

from numpy.linalg import inv
# K =  BA^(-1)
K = inv(XtX).dot(XtY) #/learning_rate
R = I - XtX*learning_rate
R = np.linalg.matrix_power(R, 1000)

# w^n = R^n(w - K) + K

for it in range(iter_time):
    # dw = learning_rate*((XtX.dot(w.reshape(137, 1)) - XtY).reshape(137, ))
    w = (R.dot(w.reshape(137, 1) - K) + K).reshape(137, )
       
    if (it+1)%100 == 0 :
        loss = YY - 2*w.dot(XtY) + w.dot(XtX).dot(w)
        print("Loss :", loss[0][0])
        print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
        result.write("%d,%f,%f\n" % (it+1, loss[0][0], mypack.cal_ndcg(w, t_list+v_list)) )

        np.savetxt("task3.model", w)

result.close()

print("Train NDCG:", mypack.cal_ndcg(w, t_list))
print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
