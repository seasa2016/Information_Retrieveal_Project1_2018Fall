#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task A
Relevant query v.s. Relevant comments
BM25
"""
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

warning_ignored = np.seterr(divide = 'ignore', invalid = 'ignore')

# 1. Import training dataset which is saved when running the file, 'Data_preprocessing.py'
train_2016 = pd.read_csv('data/csv_files/train_A.csv', encoding = 'utf-8').iloc[:, 1:]
train_2015 = pd.read_csv('data/csv_files/train_A_2015.csv', encoding = 'utf-8').iloc[:, 1:]
# val = pd.read_csv('data/csv_files/val_A.csv').iloc[:, 1:]
train = train_2016 # pd.concat([train_2016, train_2015], 0)
# test = pd.read_csv('data/csv_files/test_A.csv').iloc[:, 1:]
# final_test = pd.read_csv('data/csv_files/final_test_A.csv').iloc[:, 1:]

# train.columns.values
# ['RELQ_ID', 'RelQBody', 'RELC_ID', 'RELC_RELEVANCE2RELQ', 'RelCText']

# 2. Preprocessing
# (a) Extract texts of all queries and comments from dataset
tr_queries = np.unique(train['RelQBody'])
tr_comments = train['RelCText']
tr_q_c = np.hstack((tr_queries, tr_comments)) # merge queries and comments

# (b) Tokenize and build the vocabulary
def tokenize(alist):
    letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(alist))
    token = re.sub(r'[)!;/.?:-]', ' ', letters_only)
    token = word_tokenize(token.lower())
    return token

def qc_token(q_or_c):
    qcs = []
    for i in q_or_c:
        qcs.extend(tokenize(i))
    return np.unique(qcs)

tr_qs = [tokenize(q) for q in tr_queries] # for each query to tokenize
tr_cs = [tokenize(c) for c in tr_comments] # for each comment to tokenize
vocabulary = qc_token(tr_q_c)  # size of vocabulary = 42078, all tokenized queries and comments to be as vocabulary

# (c) Trigram and hashed vocabulary
def word_ngram(word): # input: a word
    word = '#' + word + '#'
    hash_word = []
    for i in range(len(word) - 2): # take every three characters together
        hash_word.append(word[i : (i + 3)])
    return hash_word # output: tri-letters as a word's representation 

def qc_ngram(q_or_c): # input: a query or a comment
    qcs = []
    for word in q_or_c:
        hash_word = word_ngram(word)
        qcs.extend(hash_word)    
    """qcs.append('unknown') # add the unknown-word category""" # it's not necessary to build an unknown category due to n-gram
    return qcs # output: a query or a comment represented as tri-letter words (a vector)

hash_tr_qs = [qc_ngram(q) for q in tr_qs] # transform queries to tri-letter-words vectors
hash_tr_cs = [qc_ngram(c) for c in tr_cs] # transform comments to tri-letter-words vectors
hash_vocabulary = np.unique(qc_ngram(vocabulary)) # size of hash_vocabulary = 8466
# np.save('savings/taskA/hash_vocabulary.npy', hash_vocabulary)

# (d) Term frequency (/ One-hot encoding) as the representation for queries and comments
def tf(qcs): # input: tri-letter-words vectors (representation of queries or comments)
    q_or_c_vecs = []
    for q_or_c in qcs: # for each tri-letter-words vector
        q_or_c_vec = []
        # Count how much times that each index term appears in the query or comment
        for word in hash_vocabulary:# for each tri-letter word in the tri-letter vocabulary / index terms
            q_or_c_vec.append(list(q_or_c).count(word)) # tf
            """One-hot encoding
            
            if word in q_or_c:
                q_or_c_vec.append(1)
            else:
                q_or_c_vec.append(0)
            """
        q_or_c_vecs.append(q_or_c_vec) 
    return q_or_c_vecs # output: term-frequency represented tri-letter-words vectors (queries or comments)

# tr_q_vecs = tf(hash_tr_qs) # term-frequency represented tri-letter-words vectors (queries)
# tr_c_vecs = tf(hash_tr_cs) # term-frequency represented tri-letter-words vectors (comments)

# np.save('savings/taskA/tr_q_vecs.npy', tr_q_vecs)
# np.save('savings/taskA/tr_c_vecs.npy', tr_c_vecs)

# (e) Label - ground truth that if comment is relevant or not
def formulate_label(original_label, benchmark_list, relevant_bool): # relevant or not represented by boolean or num
    if relevant_bool:
        if original_label in benchmark_list:
            new_label = 'True'
        else:
            new_label = 'False'
    else:
        if original_label in benchmark_list:
            new_label = 1
        else:
            new_label = 0
    return new_label

benchmark_list = np.unique(train['RELC_RELEVANCE2RELQ'])[1] # relevant
label = [formulate_label(i, benchmark_list, False) for i in train['RELC_RELEVANCE2RELQ']]
label_bool = [formulate_label(i, benchmark_list, True) for i in train['RELC_RELEVANCE2RELQ']]

# (f) Term frequency
tr_q_vecs = np.load('savings/taskA/tr_q_vecs.npy') # size = (5063, 10191)
tr_c_vecs = np.load('savings/taskA/tr_c_vecs.npy') # size = (49268, 10191)

# 3. BM25
# (a) Similarities definition
def bm25(q, djs, K1, b): # input: one query, comments pop up, K1, b
    N = len(djs)
    avg_len_djs = np.mean([np.sum(dj) for dj in djs]) # average length of all comments given a query
    # notice that np.sum() used is because dj is represented by term-frequency notion
    # so the sum of dj is a comment's length
    
    sim_q_djs = [] # similarities of documents with a given query
    # For each of the comments to be ranked given a query
    for j in range(N): # for each comment
        ki_idx_in_q_dj = list(set(np.where(q != 0)[0]) & set(np.where(djs[j] != 0)[0])) # index term, ki, in both query and comments popping up
        
        if not ki_idx_in_q_dj: # if there's no words in both query and comment then similarity = 0
            sim_q_dj = 0
        else:
            fij = djs[j][ki_idx_in_q_dj] # Frequency(i-th trigram in j-th comment)           
            len_dj = np.sum(djs[j]) # length of j-th comment
            
            Bij = ((K1 + 1) * fij) / (K1 * (1 - b + b * len_dj / avg_len_djs) + fij + 1e-7) # term frequency factor with consideration of the length of the comments
                                                                                            # Add 1e-7 to prevent from value = 0
            nis = [np.sum([1 for dj in djs if dj[ki_idx] != 0]) for ki_idx in ki_idx_in_q_dj] # document frequencies
            
            sim_q_dj = np.sum(Bij * np.log2((N - np.float32(nis) + 0.5) / (np.float32(nis) + 0.5))) 
        # similarity calculated based on both term frequency factor and document frequencies (tf-idf)
        sim_q_djs.append(sim_q_dj)
        
    return sim_q_djs

# (b) Similarities 
def sim_q_djs(qs, rel_qs, K1, b):
    qs_sim_q_djs = []
    for idx in range(len(qs)):
        loc = np.where(train['RelQBody'] == tr_queries[idx])[0]
        tr_rel_qs = rel_qs[loc, :]
        sim_q_djs = bm25(qs[idx], tr_rel_qs, K1, b)
        qs_sim_q_djs.append(sim_q_djs)
    return qs_sim_q_djs

"""
def sim_q_djs(qs, q_djs, K1, b, N):      
    qs_sim_q_djs = []
    for q in range(len(qs)):
        djs = q_djs[10 * q : 10 * (q + 1)]
        sim_q_djs = bm25(q, djs, K1, b, N)
        qs_sim_q_djs.append(sim_q_djs)
    return qs_sim_q_djs
"""

# (c) Transform similarities to [0, 1] 
def trans_sim(sim_q_djs):
    m = np.min(sim_q_djs)
    M = np.max(sim_q_djs)
    new_sim = (sim_q_djs - m) / (M - m)
    return new_sim

# (c) Find the best K1, b and threshold to determine whether it is more accurate
def relevance(qs, rel_qs, K1, b, threshold):
    qs_sim_q_djs = sim_q_djs(qs, rel_qs, K1, b)
    new_sims = []
    for i in qs_sim_q_djs:
        new_sims.extend(trans_sim(i)) # i
    relevance = np.zeros_like(new_sims)
    relevance[np.where(new_sims >= threshold)[0]] = 1
    return relevance
"""
def relevance(qs, q_djs, K1, b, threshold, N = 10):
    qs_sim_q_djs = sim_q_djs(qs, q_djs, K1, b, N)
    new_sims = np.array([trans_sim(i) for i in qs_sim_q_djs]).reshape([q_djs.shape[0], 1])
    relevance = np.zeros_like(new_sims)
    relevance[np.where(new_sims >= threshold)[0]] = 1
    return relevance
"""
def accuracy(relevance_pred, relevance_true):
    return np.mean(relevance_pred == relevance_true)

K1 = [0.5, 1., 1.5] # smooth factor
b = [0.7, 0.85, 1.] # b will be set as a value closed to 1 if you want to put emphasis on the length of a document
threshold = [0.55, 0.75, 0.95] # threshold that those transformed similarities that are larger than it will be determined as relevant comments

hyper_param = np.vstack((np.vstack((np.repeat(K1, 9),
                                    np.tile(np.repeat(b, 3), 3))),
                         np.tile(threshold, 9))).T
""" Hyper-parameters combinations

        K1     b  threshold
array([[0.5 , 0.7 , 0.55],
       [0.5 , 0.7 , 0.75],
       [0.5 , 0.7 , 0.95],
       [0.5 , 0.85, 0.55],
       [0.5 , 0.85, 0.75],
       [0.5 , 0.85, 0.95],
       [0.5 , 1.  , 0.55],
       [0.5 , 1.  , 0.75],
       [0.5 , 1.  , 0.95],
       [1.  , 0.7 , 0.55],
       [1.  , 0.7 , 0.75],
       [1.  , 0.7 , 0.95],
       [1.  , 0.85, 0.55],
       [1.  , 0.85, 0.75],
       [1.  , 0.85, 0.95],
       [1.  , 1.  , 0.55],
       [1.  , 1.  , 0.75],
       [1.  , 1.  , 0.95],
       [1.5 , 0.7 , 0.55],
       [1.5 , 0.7 , 0.75],
       [1.5 , 0.7 , 0.95],
       [1.5 , 0.85, 0.55],
       [1.5 , 0.85, 0.75],
       [1.5 , 0.85, 0.95],
       [1.5 , 1.  , 0.55],
       [1.5 , 1.  , 0.75],
       [1.5 , 1.  , 0.95]])
"""
train_accs = []
for param in hyper_param:
    rel = relevance(tr_q_vecs, tr_c_vecs, param[0], param[1], param[2])
    train_acc = accuracy(rel, label)
    train_accs.append(train_acc)
    
""" Training accuracies based on different combinations of hyper-parameters

[0.554304381245196,
 0.5843966179861645,
 0.6036510376633359,
 0.5529976940814758,
 0.5831667947732514,
 0.6031898539584934,
 0.5510376633358954,
 0.5814373558800923,
 0.603228285933897,
 0.5523059185242122,
 0.5823597232897771,
 0.6029976940814757,
 0.548923904688701,
 0.5824750192159877,
 0.603228285933897,
 0.5455418908531898,
 0.5833205226748654,
 0.6045349730976172,
 0.5507686395080708,
 0.5835511145272867,
 0.6024212144504227,
 0.5463489623366641,
 0.5842428900845503,
 0.6036126056879324,
 0.5440046118370484,
 0.5818985395849346,
 0.6039584934665642]
"""
highest_train_acc = np.max(train_accs) # 0.60453
best_params = hyper_param[np.argmax(train_accs)] # [1.  , 1.  , 0.95]
# np.save('savings/taskA/best_params.npy', best_params)


