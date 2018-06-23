import semanticanalysis as sa
from datetime import datetime

class Articles(sa.Documents):
    def __init__(self, fname):
        super().__init__(
            fname=fname, 
            tabname='articles', 
            colschema='id integer primary key autoincrement, src string, date int, datestr string, num integer, fname string, text string, sents blob, entities blob',
        )
        self.c.execute("create index if not exists idx1 on articles(src)")
        self.c.execute("create index if not exists idx2 on articles(date)")
        #self.c.execute("create index if not exists idx3 on articles(datestr)")
        self.c.execute("create index if not exists idx4 on articles(src,date)")
        #self.c.execute("create index if not exists idx5 on articles(src,datestr)")
        self.c.execute("create index if not exists idx6 on articles(date, src)")
        #self.c.execute("create index if not exists idx7 on articles(datestr, src)")
        
class WTVDB(sa.Documents):
    def __init__(self, fname):
        super().__init__(
            fname=fname, 
            tabname='wtvmodels', 
            colschema='id integer primary key autoincrement, src string, ndim int, window int, min_count int, model blob',
        )
