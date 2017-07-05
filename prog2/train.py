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

learning_rate = 1/qc
iter_time = 2000
alpha = 0.00001

while True:
    w = np.random.rand(137)*2 - 1
    v = mypack.cal_ndcg(w, v_list)
    print("Cal V", v)
    if v > 0.275 : break

print("Choose w with ndcg in valid data =", v)

pool = Pool(8)

tp_list = []

max_doc_list = 0
for qid, doc_list in t_list:
    rg = range(len(doc_list))
    max_doc_list = max(max_doc_list, len(doc_list))
    p_list = []
    for _ in range(int(len(doc_list))):
        i, j = random.choice(rg), random.choice(rg)
        x1, x2 = doc_list[i], doc_list[j]
        if x1[0] <= x2[0]: continue
        # rel of x1 > rel of x2
        p_list.append((i, j))
    tp_list.append(list(set(p_list)))

print("Pair list fixed.")

tc = len(t_list)
ind_list = list(range(tc))
batch_count = 10
batch_len = tc//batch_count

dw = 0

result = open("task1-result.csv", "w")
result.write("Iteration,Loss,Train Data NDCG\n")

for it in range(iter_time):
    batch_id = it%batch_count
    batch_ind = ind_list[batch_id*batch_len : (batch_id+1)*batch_len]

    batch_t = [t_list[i] for i in batch_ind]
    batch_tp = [tp_list[i] for i in batch_ind]
    map_result = pool.map(iters.task1, zip(itertools.cycle([w]), batch_t, batch_tp))
    dw = -learning_rate*sum(map(lambda x : x[0], map_result)) - alpha*w
    loss = sum(map(lambda x : x[1], map_result))
    w += dw

    print("Iter%d" % (it+1), ":", loss*batch_count, "\r", end="")

    if (it+1)%10 == 0:
        print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
        result.write("%d,%f,%f\n" % (it+1, loss, mypack.cal_ndcg(w, t_list+v_list)) )

np.savetxt("task1.model", w)
result.close()

print("Train NDCG:", mypack.cal_ndcg(w, t_list))
print("Valid NDCG:", mypack.cal_ndcg(w, v_list))
