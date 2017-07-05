import numpy as np
import math

import mypack

import pickle
query_map = pickle.load(open("/tmp2/b02902029/train.test.pickle", "rb"))
print("Read file ok.")

def dcg(score_list):
    ret = 0.
    for ind, (_, rel) in enumerate(score_list):
        normalizer = 1.
        if ind :
            normalizer = math.log(ind+1)/math.log(2)
        ret += rel/normalizer
    return ret

def cut(l, n):
    if len(l) > n:
        return l[:n]
    else:
        return l

w = np.loadtxt("task1.model")

avg_ndcg = 0

for qid, doc_list in query_map.items():
    score_list = [
        (np.dot(w, arr), rel)
        for rel, arr, docid in doc_list
    ]

    score_list.sort(key=lambda x : x[1], reverse=True)
    normalizer = dcg(cut(score_list, 10))

    if (not score_list) or score_list[0][1] == 0 :
        continue

    score_list.sort(key=lambda x : x[0], reverse=True)

    cal_ndcg = dcg(cut(score_list, 10))/normalizer
    avg_ndcg += cal_ndcg

print(avg_ndcg/len(query_map.items()))
