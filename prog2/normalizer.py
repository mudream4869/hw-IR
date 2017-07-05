import numpy as np
import math

import pickle
query_map = pickle.load(open("/tmp2/b02902029/train.pickle", "rb"))

print("Read file ok.")

w_normal = []

for ind in range(137):
    max_v, min_v = -math.inf, math.inf

    for qid, doc_list in query_map.items():
        for rel, arr in doc_list:
            max_v = max(max_v, arr[ind])
            min_v = min(min_v, arr[ind])

    delta = max(max_v - min_v, 1.)

    w_normal.append(delta)

np.savetxt("nor.vec", w_normal)
