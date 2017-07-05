import numpy as np

import mypack
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-task', dest="task_id")
args = parser.parse_args()

task_id = int(args.task_id)

w = np.loadtxt("task%d.model" % (task_id, ))

query_map = pickle.load(open("/tmp2/b02902029/train.test.normalize.pickle", "rb"))
print("Read normalized file ok.")

t_list, v_list = mypack.split_tv(query_map)

print("Train NDCG:", mypack.cal_ndcg(w, t_list + v_list))
