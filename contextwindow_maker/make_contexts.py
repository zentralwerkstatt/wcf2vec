
# coding: utf-8

# In[1]:


from articles import Articles
import semanticanalysis as sa
import spacy
import scipy
import numpy as np
import time


# In[2]:


# settings
WINDOW_SIZE = 5
TESTDATA_SIZE = None
PMI_ALPHA = 1.0
print('window size', WINDOW_SIZE, 'TEST DATA SIZE', TESTDATA_SIZE)


# # Get Test Dataset

# In[3]:


print('getting test data')
t0 = time.time()
docs = Articles('articles_tokenized.db')


# In[4]:


docs.getdf(limit=1)


# In[5]:


doctexts = list(docs.getdf(sel=['text',], limit=TESTDATA_SIZE)['text'])
print('took', int((time.time()-t0)/60), 'minutes')


# # Parse with spacy and build Gensim dictionary object

# In[6]:


print('making gensim dictionary')
t0 = time.time()

def parsethread(doctext,nlp):
    parsed = nlp(doctext)
    return [w.text for w in parsed if w.is_alpha and not w.is_digit]
        
nlp = spacy.load('en', disable=['ner', 'tagger', 'parser'])
sents = sa.parmap(parsethread, doctexts,nlp, workers=10)


# In[7]:


import math
from multiprocessing import Pool
import gensim

def dictbuilder_thread(docs):
    return gensim.corpora.dictionary.Dictionary(docs)

def bow2dict(bowtexts, min_tf=2, ngrams=1, workers=8, verbose=False):
    chunksize = math.ceil(len(bowtexts)/workers)
    #dictbuilder_thread = lambda docs: gensim.corpora.dictionary.Dictionary(docs)
    with Pool(workers) as p:
        dcts = p.map(dictbuilder_thread, [bowtexts[i*chunksize:(i+1)*chunksize] for i in range(workers)])
    
    if verbose: print('merging dictionaries')
    # merge them back together
    dct = dcts[0]
    for i in range(1,len(dcts)):
        dct.merge_with(dcts[i])
    
    #dct.filter_extremes(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None)
    dct.filter_extremes(no_below=min_tf, no_above=1.0, keep_n=10000000000)
    
    return dct

# (does this in parallel bc docs are kind of big)
dct = bow2dict(sents, min_tf=100, ngrams=1, workers=8, verbose=False)
print('took', int((time.time()-t0)/60), 'minutes')
len(dct)


# ## Break Up Docs/Sents Into Context Windows
# _sents_ is a list of document tokens as strings, _dct_ is a [gensim dictionary object](https://radimrehurek.com/gensim/corpora/dictionary.html).

# In[8]:


print('making context windows')
t0 = time.time()

def getcontexts(sent,argdata):
    win, dct = argdata[0], argdata[1]
    sent = [w for w in sent if w in dct.token2id.keys()]
    contexts = list()
    words = list()
    for i in range(len(sent)):
    #for i in range(win,len(sent)-1-win):
        words.append(sent[i])
        cont = list()
        for j in range(max(0,i-win),min(len(sent),i+win+1)):
            cont.append(sent[j])
        contexts.append(cont)
    return [(w,c) for w,c in zip(words, contexts)]
    
contexts = [ct for cts in sa.parmap(getcontexts, sents, (WINDOW_SIZE,dct)) for ct in cts]
#contexts = [ct for s in sents for ct in getcontexts(s,WINDOW_SIZE)]
fullcontexts = [cont for t,cont in contexts if len(cont) == WINDOW_SIZE*2+1]
print('took', int((time.time()-t0)/60), 'minutes')

fullcontexts[0]


# # Save Contexts as Matrix for Word2Vec

# In[9]:


print('saving contexts as matrix for word2vec')
t0 = time.time()

# save context windows as matrix
# for w2v need to include only contexts of window size (omit edge cases - beginning and ends of sents)
import pickle
contextmat = np.array([[dct.token2id[w] for w in cont] for t,cont in contexts if len(cont) == WINDOW_SIZE*2+1])
targetwords = np.array([dct.token2id[t] for t,cont in contexts if len(cont) == WINDOW_SIZE*2+1])
with open('contexdata.pic', 'wb') as f:
    pickle.dump(dict(contextmat=contextmat, targetwords=targetwords, id2token={i:t for t,i in dct.token2id.items()}), f)


# In[10]:


with open('contexdata.pic', 'rb') as f:
    dat = pickle.load(f)
print('took', int((time.time()-t0)/60), 'minutes')
dat['contextmat'].shape, dat['targetwords'].shape, len(dat['id2token'])


# # Make PMI Matrix

# In[11]:


print('making context-word matrix')
t0 = time.time()

def context2cwm(contexts,dct):
    C = scipy.sparse.lil_matrix((len(contexts),len(dct)), dtype=np.float64, copy=False)
    #dok_matrix((5, 5), dtype=np.float32)
    for i,cont in enumerate(contexts):
        if i % 10000 == 0: print(i)
        for w in cont:
            C[i,dct.token2id[w]] += 1
    return C

# Make content-word matrix (<num contexts> X <vocab size>)
#fullcontexts = [cont for t,cont in contexts if len(cont) == WINDOW_SIZE*2+1]
C = context2cwm(fullcontexts,dct)
print('took', int((time.time()-t0)/60), 'minutes')
C.shape


# In[12]:


print('making pmi matrix')
t0 = time.time()

def cwm2pmi(C,a=1):
    print('making pcw')
    pcw = C/C.sum().sum()
    pw = pcw.sum(axis=0)
    pc = pcw.sum(axis=1)*a
    
    print('making pmi')
    PMI = pcw/(pc.dot(pw))

    return PMI

C = scipy.sparse.csc_matrix(C)
pmi = cwm2pmi(C,PMI_ALPHA)
print('took', int((time.time()-t0)/60), 'minutes')
pmi.shape


# In[ ]:


pmi[:,dct.token2id['the']].shape


# In[ ]:


print('making svd')
t0 = time.time()
u, s, vt = scipy.sparse.linalg.svds(pmi.T, k=300, return_singular_vectors=True)
print('took', int((time.time()-t0)/60), 'minutes')


# In[ ]:


with open('pmidata.pic', 'wb') as f:
    pickle.dump(dict(vt=vt,u=u,s=s), f)

