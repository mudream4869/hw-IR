

def read_map(fn):
    fp = open(fn)
    fp.readline()
    ret_map = {}
    for line in fp:
        row = line.split(",")
        rank_map = {}
        ind = 0
        for f in row[1].split():
            rank_map[f] = ind
            ind += 1
        ret_map[row[0]] = (rank_map, row[1].split())

    return ret_map


def map_grade(out_file, ans_file):
    out_map = read_map(out_file)
    ans_map = read_map(ans_file)

    score_sum = 0
    score_count = 0

    for k, v in ans_map.items():
        num = []
        for ind, fn in enumerate(v[1]):
            if fn in out_map[k][0]:
                num.append(out_map[k][0][fn]+1)
        
        num.sort()
        acc = 0
        for i, val in enumerate(num):
            acc += (i+1)/val

        acc /= len(v[1])
        score_sum += acc
        score_count += 1

        print(k, acc)

    return score_sum/score_count


query_dir = "/tmp/b02902029/ir-hw1/queries/"
print(map_grade("output-train.csv","/tmp/b02902029/ir-hw1/queries/ans_train.csv"))
