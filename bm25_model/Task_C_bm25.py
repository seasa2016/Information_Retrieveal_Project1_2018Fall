#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task C
Original query v.s. Relevant comments
BM25
"""
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize


# 1. Import training dataset which is saved when running the file, 'Data_preprocessing.py'
train = pd.read_csv('data/csv_files/train_C.csv', encoding = 'utf-8').iloc[:, 1:]
# val = pd.read_csv('data/csv_files/val_C.csv').iloc[:, 1:]
# test = pd.read_csv('data/csv_files/test_C.csv').iloc[:, 1:]
# final_test = pd.read_csv('data/csv_files/final_test_C.csv').iloc[:, 1:]

# train.columns.values
# ['ORGQ_ID', 'OrgQBody', 'RELC_ID', 'RELC_RELEVANCE2RELQ', 'RelCText']


# 2. Preprocessing
# (a) Extract texts of all queries and comments from dataset
tr_queries = np.unique(train['OrgQBody'])
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

tr_qs = [tokenize(q) for q in tr_queries]
tr_cs = [tokenize(c) for c in tr_comments]
vocabulary = qc_token(tr_q_c)  # size of vocabulary = 26782

# (c) Trigram and hashed vocabulary
def word_ngram(word):
    word = '#' + word + '#'
    hash_word = []
    for i in range(len(word) - 2):
        hash_word.append(word[i : (i + 3)])
    return hash_word

def qc_ngram(q_or_c):
    qcs = []
    for word in q_or_c:
        hash_word = word_ngram(word)
        qcs.extend(hash_word)    
    """qcs.append('unknown') # add the unknown-word category"""
    return qcs

hash_tr_qs = [qc_ngram(q) for q in tr_qs]
hash_tr_cs = [qc_ngram(c) for c in tr_cs]
hash_vocabulary = np.unique(qc_ngram(vocabulary)) # size of hash_vocabulary = 8333
# np.save('savings/taskC/taskC_hash_vocabulary.npy', hash_vocabulary)

# (d) Term frequency / One-hot encoding as the representation for queries and comments
def tf(qcs):
    q_or_c_vecs = []
    for q_or_c in qcs:
        q_or_c_vec = []
        for word in hash_vocabulary:
            q_or_c_vec.append(list(q_or_c).count(word)) # tf
            """One-hot encoding
            
            if word in q_or_c:
                q_or_c_vec.append(1)
            else:
                q_or_c_vec.append(0)
            """
        q_or_c_vecs.append(q_or_c_vec) 
    return q_or_c_vecs

# tr_q_vecs = tf(hash_tr_qs)
# tr_c_vecs = tf(hash_tr_cs)
#
# np.save('savings/taskC/taskC_tr_q_vecs.npy', tr_q_vecs)
# np.save('savings/taskC/taskC_tr_c_vecs.npy', tr_c_vecs)

# (e) Label
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

benchmark_list = np.unique(train['RELC_RELEVANCE2RELQ'])[1]
label = [formulate_label(i, benchmark_list, False) for i in train['RELC_RELEVANCE2RELQ']]
label_bool = [formulate_label(i, benchmark_list, True) for i in train['RELC_RELEVANCE2RELQ']]

# (f) Term frequency
tr_q_vecs = np.load('savings/taskC/taskC_tr_q_vecs.npy') # size = (2669, 8333)
tr_c_vecs = np.load('savings/taskC/taskC_tr_c_vecs.npy') # size = (26690, 8333)

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
        loc = np.where(train['OrgQBody'] == tr_queries[idx])[0]
        tr_rel_qs = rel_qs[loc, :]
        sim_q_djs = bm25(qs[idx], tr_rel_qs, K1, b)
        qs_sim_q_djs.append(sim_q_djs)
    return qs_sim_q_djs



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
        new_sims.extend(trans_sim(i))
    relevance = np.zeros_like(new_sims)
    relevance[np.where(new_sims >= threshold)[0]] = 1
    return relevance

def accuracy(relevance_pred, relevance_true):
    return np.mean(relevance_pred == relevance_true)


K1 = [0.2, 0.5, 1.] # smooth factor
b = [0.7, 0.85, 1.] # b will be set as a value closed to 1 if you want to put emphasis on the length of a document
threshold = [0.55, 0.75, 0.95] # threshold that those transformed similarities that are larger than it will be determined as relevant comments

hyper_param = np.vstack((np.vstack((np.repeat(K1, 9),
                                    np.tile(np.repeat(b, 3), 3))),
                         np.tile(threshold, 9))).T
"""Hyper-parameters combinations

        K1     b  threshold
array([[0.2 , 0.7 , 0.55],
       [0.2 , 0.7 , 0.75],
       [0.2 , 0.7 , 0.95],
       [0.2 , 0.85, 0.55],
       [0.2 , 0.85, 0.75],
       [0.2 , 0.85, 0.95],
       [0.2 , 1.  , 0.55],
       [0.2 , 1.  , 0.75],
       [0.2 , 1.  , 0.95],
       [0.5 , 0.7 , 0.55],
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
       [1.  , 1.  , 0.95]])
"""
train_accs = []
for param in hyper_param:
    rel = relevance(tr_q_vecs, tr_c_vecs, param[0], param[1], param[2])
    train_acc = accuracy(rel, label)
    train_accs.append(train_acc)
    
"""Training accuracies based on different combinations of hyper-parameters

[0.5935931060322218,
 0.6149494192581492,
 0.6237542150618209,
 0.5915324091420008,
 0.614275009366804,
 0.6237542150618209,
 0.5905957287373548,
 0.6133383289621581,
 0.6233795428999626,
 0.5862495316597977,
 0.6122517796927688,
 0.6235294117647059,
 0.5848257774447359,
 0.6115399025852379,
 0.6234544773323342,
 0.5817159985013114,
 0.6119895091794679,
 0.6236418134132634,
 0.5832521543649307,
 0.6116897714499813,
 0.6238666167103785,
 0.5793555638816036,
 0.6107156238291495,
 0.6231172723866617,
 0.5771075309104533,
 0.6103034844511053,
 0.6227426002248033]
"""
highest_train_acc = np.max(train_accs) # 0.6239
best_params = hyper_param[np.argmax(train_accs)] # [1.  , 0.7 , 0.95]
# np.save('savings/taskC/taskC_best_params.npy', best_params)