import sqlite3
import math

import os.path
import build

# utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)


class Model:
    def __init__(self, model_dir):
        print("Checking model...\n", end="")

        if os.path.isfile(model_dir + "/inv-table.db") == False:
            build.build_db(model_dir)

        print("Check model ok.")

        print("Reading model...\r", end="")

        # read encode
        with open(model_dir + "/vocab.all", "r") as encode_tester: 
            enc = encode_tester.readline()[:-1]
        
        with open(model_dir + "/vocab.all", "r", encoding=enc) as f:
            self.vocab2id = {}
            f.readline() # skip "utf-8"
            ind = 1
            for vocab in f:
                self.vocab2id[vocab[:-1]] = ind
                ind += 1

        with open(model_dir + "/file-list", "r") as doc_file:
            self.file_list = []
            for line in doc_file:
                self.file_list.append(line[:-1].split("/")[-1].lower())

        self.D = len(self.file_list)

        self.db = sqlite3.connect(model_dir + "/inv-table.db")

        sqlstr = ('SELECT DOCID, LEN FROM DOCLEN') 
        cur = self.db.execute(sqlstr)

        self.file_len = [0]*self.D
        self.doclen_avg = 0
        for row in cur:
            self.file_len[row[0]] = row[1]
            self.doclen_avg += row[1]

        self.doclen_avg /= self.D

        sqlstr = ('SELECT ID, DF FROM TERM')
        cur = self.db.execute(sqlstr)

        self.idf_map = {row[0] : math.log(self.D/(row[1]+0.5)) for row in cur}

        print("Read model OK.")

    def randomFile(self):
        import random
        return random.choice(self.file_list)
    
    def getTermID(self, d):
        sqlstr = ('SELECT ID FROM TERM WHERE VOCAB1 = %s AND VOCAB2 = %s')
        sqlarr = (d[0], d[1])
        cur = self.db.execute(sqlstr % sqlarr)
        row = cur.fetchone()
        if row:
            return int(row[0])
        else:
            return 0

    def getTF(self, f, docid):
        b = 0.75
        k = 3
        pvt = self.file_len[docid]/self.doclen_avg
        return (1+k)*f/(f+k*(1-b+b*pvt))
    
    def getF(self, docid, termid):
        sqlstr = ('SELECT TIME FROM INV WHERE ID = %s AND DOCID = %s')
        sqlarr = (termid, docid)
        cur = self.db.execute(sqlstr % sqlarr)
        row = cur.fetchone()
        if row:
            return int(row[0])
        else:
            return 0

    def getFmap(self, termid):
        sqlstr = ('SELECT DOCID, TIME FROM INV WHERE ID = %s')
        sqlarr = (termid, )
        cur = self.db.execute(sqlstr % sqlarr)
        return {row[0] : row[1] for row in cur.fetchall()}

    def getIDF(self, termid):
        sqlstr = ('SELECT DOCID FROM INV WHERE ID = %s')
        sqlarr = (termid, )
        cur = self.db.execute(sqlstr % sqlarr)

        K = len(cur.fetchall())

        return math.log(self.D/(K+0.5))
    
    def getTermidList(self, gram_list):
        termid_list = []
        valid_list = []
        for gram in gram_list:
            termid = 0
            if len(gram) == 2 :
                termid = self.getTermID((self.vocab2id[gram[0]], self.vocab2id[gram[1]]))
            elif len(gram) == 1:
                termid = self.getTermID((self.vocab2id[gram[0]], -1))

            if termid: 
                termid_list.append(termid)
                valid_list.append(gram)

        return termid_list, gram_list
    
    def getFullDoc(self, docid):
        sqlstr = ('SELECT ID, TIME FROM INV WHERE DOCID = %s')
        sqlarr = (docid, )
        cur = self.db.execute(sqlstr % sqlarr)

        termid_list = []
        doc_vec = []
        for row in cur:
            termid_list.append(row[0])
            doc_vec.append(self.getTF(row[1], docid))

        idf_map = self.idf_map #{termid : self.getIDF(termid) for termid in termid_list}

        for ind in range(len(doc_vec)):
            termid = termid_list[ind]
            doc_vec[ind] *= idf_map[termid]
        
        return termid_list, doc_vec

    def getDocSim(self, qv, termid_list):

        idf_map = self.idf_map #{termid : self.getIDF(termid) for termid in termid_list}
        f_map = { termid : self.getFmap(termid) for termid in termid_list}
        def f_func(docid, termid):
            if docid not in f_map[termid]:
                return 0
            return f_map[termid][docid]

        doc_sim_list = []


        for docid in range(len(self.file_list)):
            print( "%d/%d       \r" % (docid+1, len(self.file_list)), end="")
            dv = [
                self.getTF(f_func(docid, termid), docid)*idf_map[termid]
                for termid in termid_list
            ]
            doc_sim_list.append(sim(dv, qv))

        return doc_sim_list
    
    def concat(self, termid_list1, vec1, termid_list2, vec2, alpha):

        termid_map = {v : i for i, v in enumerate(termid_list1)}

        len2 = lenv(vec2)

        for i, v in enumerate(termid_list2):
            if v in termid_map:
                ind = termid_map[v]
                vec1[ind] += vec2[i]*alpha
            else:
                if vec2[i]/len2 < 1 : continue
                termid_map[v] = len(termid_list1)
                termid_list1.append(v)
                vec1.append(vec2[i]*alpha)

        return termid_list1, vec1


def lenv(v):
    return sum(map(lambda x : x**2, v))**0.5 or 1


def sim(v1, v2):
    dim = len(v1)
    
    len_v1, len_v2, mul = 0, 0, 0
    for i in range(dim):
        len_v1 += v1[i]**2
        len_v2 += v2[i]**2
        mul += v1[i]*v2[i]

    if len_v1*len_v2 == 0:
        return -100000000 

    return mul/(len_v1*len_v2)**0.5


def mul(v, a):
    return [i*a for i in v]


def dealQuery(query_file, output_file, model, rocchio=False):
    import xml.etree.ElementTree
    qroot = xml.etree.ElementTree.parse(query_file).getroot()
    output = open(output_file, "w")
    output.write("query_id,retrieved_docs\n")
    for elem in qroot.findall("topic"):
        number_str = elem.findall("number")[0].text[-3:]

        # Get gram list

        concept_str = elem.findall("concepts")[0].text
        narrative_str = elem.findall("narrative")[0].text
        title_str = elem.findall("title")[0].text

        gram_list = []

        for gram in concept_str.split("、"):
            gram = gram.replace("\n", "").replace("。", "").replace("\n", "").strip()
            if len(gram) == 1:
                gram_list.append(gram)
            elif len(gram) == 2:
                gram_list.append(gram)
                gram_list.append(gram[0])
                gram_list.append(gram[1])
            else:
                for ind in range(len(gram) - 1):
                    gram_list.append(gram[ind:ind+2])


        termid_list, gram_list = model.getTermidList(list(set(gram_list)))
        idf_map = model.idf_map 
        
        # Query vector

        qv = [ idf_map[termid] for i, termid in enumerate(termid_list) ]

        for sentence in narrative_str.split("。"):
            if sentence.count("不相關"):
                for ind, gram in enumerate(gram_list):
                    if len(gram) == 2 and sentence.count(gram):
                        qv[ind] *= 1.5

        # Get doc sim and sort

        dv_list = model.getDocSim(qv, termid_list)
        dv_score = [(s, ind) for ind, s in enumerate(dv_list)]
        dv_score.sort(key=lambda t : t[0], reverse=True)

        # Rocchio Feedback

        if rocchio:

            feedback_pos = 20
            print("Feedback Doc %d/%d     \r" % (0, feedback_pos), end="")
            for i in range(feedback_pos):
                termid_list2, doc_vec = model.getFullDoc(dv_score[i][1])
                termid_list, qv = model.concat(termid_list, qv, termid_list2, doc_vec, 1/feedback_pos * 0.05)
                print("Feedback Doc %d/%d     \r" % (i+1, feedback_pos), end="")

            print("After Feedback : dim of query vector =", len(qv))

            dv_list = model.getDocSim(qv, termid_list)
            dv_score = [(s, ind) for ind, s in enumerate(dv_list)]
            dv_score.sort(key=lambda t : t[0], reverse=True)

        # List 100 files

        file_list = [ model.file_list[pr[1]] for pr in dv_score[:100] ]

        output.write("%s,%s\n" % (number_str, " ".join(file_list)))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', help='Turn on relevance feedback (Rocchio algorithm).', action='store_true', dest="rocchio")
    parser.add_argument('-i', help='The input query file', dest="query_file")
    parser.add_argument('-m', help='The model directory, which includes : vocab.all, file-list and inverted-index', dest="model_dir")
    parser.add_argument('-d', help='NTCIR directory', dest="ntcir_dir")
    parser.add_argument('-o', help='The output ranked list file', dest="output_file")
    args = parser.parse_args()

    query_file = args.query_file    # "/tmp/b02902029/ir-hw1/queries/query-train.xml"
    model_dir = args.model_dir      # "/tmp/b02902029/ir-hw1/model"
    output_file = args.output_file  # "output.csv"
    rocchio = args.rocchio

    model = Model(model_dir)

    dealQuery(query_file, output_file, model, rocchio)
