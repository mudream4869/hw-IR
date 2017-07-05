import numpy as np
import math

import mypack


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-task', dest="task_id")
parser.add_argument('-input', dest="input")
parser.add_argument('-output', dest="output")
args = parser.parse_args()

task_id = int(args.task_id)
input_file = args.input
output_file = args.output

query_map = mypack.readfile(input_file, train=False)
print("Read file ok.")

output = open(output_file, "w")
output.write("QueryId,DocumentId\n")
w = np.loadtxt("src/task%d.model" % (task_id, ))/np.loadtxt("src/nor.vec")

for qid, doc_list in query_map.items():
    score_list = [
        (np.dot(w, arr), docid)
        for rel, arr, docid in doc_list
    ]

    score_list.sort(key=lambda x : x[0], reverse=True)
    print(list(map(lambda x : x[1], score_list[:10])))

    for score, docid in score_list[:10]:
        output.write("%d,%d\n" % (qid, docid))

output.close()
