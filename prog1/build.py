import sqlite3

def build_db(model_dir):
    conn = sqlite3.connect(model_dir + "/inv-table.db")
    conn.execute(
    '''
        CREATE TABLE TERM(
            ID      INT PRIMARY KEY,
            VOCAB1  INT, 
            VOCAB2  INT,
            DF      INT,
            UNIQUE(VOCAB1, VOCAB2)
        )
    '''
    )
    conn.execute(
    '''
        CREATE TABLE INV(
            ID      INT,
            DOCID   INT, 
            TIME    INT
        )
    '''
    )
    conn.execute(
    '''
        CREATE TABLE DOCLEN(
            DOCID   INT, 
            LEN     INT
        )
    '''
    )
    conn.execute("CREATE INDEX INV_INDEX_ID ON INV (ID)")
    conn.execute("CREATE INDEX INV_INDEX_DOCID ON INV (DOCID)")
    conn.execute("CREATE UNIQUE INDEX TERM_INDEX_VOC ON TERM (VOCAB1, VOCAB2)")
    conn.execute("CREATE UNIQUE INDEX DOCLEN_INDEX_DOCID ON DOCLEN (DOCID)")
    with open(model_dir + "/inverted-file") as inv_file: 
        
        term_id = 1

        acc = 0
        pc_line = 0

        doclen_map = {}

        cur = conn.cursor()

        while True:
            line = inv_file.readline()
            if not line : break
            voc1, voc2, n = map(int, line.split())
            conn.execute("INSERT INTO TERM(ID, VOCAB1, VOCAB2, DF) VALUES (%d, %d, %d, %d)" % (term_id, voc1, voc2, n))
            for _ in range(n):
                docid, time = map(int, inv_file.readline().split())
                conn.execute("INSERT INTO INV(ID, DOCID, TIME) VALUES (%d, %d, %d)" % (term_id, docid, time))

                if docid in doclen_map : 
                    doclen_map[docid] += time
                else:
                    doclen_map[docid] = time


            acc += n+1
            pc_line += n+1

            if acc >= 2000000 :
                conn.commit()
                acc = 0

            term_id += 1
            print(" \r%d/100 => term : %d, line : %d/37320537 " % (int(pc_line*100/37320537.0), term_id, pc_line), end="")

        conn.commit()
        
        for docid, dlen in doclen_map.items():
            conn.execute("INSERT INTO DOCLEN (DOCID, LEN) VALUES (%s, %s)" % (docid, dlen))

        conn.commit()

        print("")



if __name__ == "__main__":
    # Test code
    model_dir = "/tmp/b02902029/ir-hw1/model"
    build_db(model_dir)
