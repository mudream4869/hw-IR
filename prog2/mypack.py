import numpy as np
import math
import random

def readfile(filename, line_limit=math.inf, train=True):
    train_file = open(filename, "r")
    query_map = {}
    for lc, line in enumerate(train_file):
        if lc > line_limit: break
        sp_line = line.split()
        rel = int(sp_line[0])
        qid = int(sp_line[1].split(":")[1])
        docid = int(sp_line[2].split(":")[1])
        arr = np.array(list(map(
                lambda x : float(x.split(":")[1]),
                sp_line[3:]
              )) + [1.])

        if qid not in query_map :
            query_map[qid] = []
        
        if train:
            query_map[qid].append((rel, arr))
        else:
            query_map[qid].append((rel, arr, docid))

    train_file.close()
    
    return query_map


def split_tv(query_map):
    train_list = []
    valid_list = []
    for _item in query_map.items():
        if random.randrange(5):
            train_list.append(_item)
        else:
            valid_list.append(_item)

    return train_list, valid_list


def dcg(score_list):
    ret = 0.
    for ind, (_, rel) in enumerate(score_list):
        normalizer = math.log(ind+2)/math.log(2)
        ret += (2**rel-1)/normalizer
    return ret


def cal_ndcg(w, q_list):
    avg_ndcg = 0

    for qid, doc_list in q_list:
        score_list = [
            (np.dot(w, arr), rel)
            for rel, arr, docid in doc_list
        ]
        
        if not score_list :
            continue

        score_list.sort(key=lambda x : x[1], reverse=True)
        normalizer = dcg(score_list[:10])

        if score_list[0][1] == 0 :
            continue

        random.shuffle(score_list)
        score_list.sort(key=lambda x : x[0], reverse=True)

        avg_ndcg += dcg(score_list[:10])/normalizer 

    return avg_ndcg/len(q_list)
