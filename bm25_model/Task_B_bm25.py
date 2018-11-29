#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task B
Original query v.s. Relevant query
BM25
"""

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

# 1. Import training dataset which is saved when running the file, 'Data_preprocessing.py'
train = pd.read_csv('data/csv_files/train_B.csv').iloc[:, 1:]
# val = pd.read_csv('data/csv_files/val_B.csv').iloc[:, 1:]
# test = pd.read_csv('data/csv_files/test_B.csv').iloc[:, 1:]
# final_test = pd.read_csv('data/csv_files/final_test_B.csv').iloc[:, 1:]

# train.columns.values
# ['ORGQ_ID', 'OrgQBody', 'RELQ_ID', 'RelQBody', 'RELQ_RELEVANCE2ORGQ']


# 2. Preprocessing
# (a) Extract texts of all queries from dataset
tr_org_queries = np.unique(train['OrgQBody'])
tr_rel_queries = train['RelQBody']
tr_q_q = np.hstack((tr_org_queries, tr_rel_queries)) # merge all queries

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

tr_org_qs = [tokenize(q) for q in tr_org_queries]
tr_rel_qs = [tokenize(c) for c in tr_rel_queries]
vocabulary = qc_token(tr_q_q)  # size of vocabulary = 7386

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

hash_tr_org_qs = [qc_ngram(q) for q in tr_org_qs]
hash_tr_rel_qs = [qc_ngram(q) for q in tr_rel_qs]
hash_vocabulary = np.unique(qc_ngram(vocabulary)) # size of hash_vocabulary = 4827
# np.save('savings/taskB/taskB_hash_vocabulary.npy', hash_vocabulary)

# (d) Term frequency / One-hot encoding as the representation for queries
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

# tr_org_q_vecs = tf(hash_tr_org_qs)
# tr_rel_q_vecs = tf(hash_tr_rel_qs)

# np.save('savings/taskB/tr_org_q_vecs.npy', tr_org_q_vecs)
# np.save('savings/taskB/tr_rel_q_vecs.npy', tr_rel_q_vecs)

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

benchmark_list = np.unique(train['RELQ_RELEVANCE2ORGQ'])[1:]
label = [formulate_label(i, benchmark_list, False) for i in train['RELQ_RELEVANCE2ORGQ']]
label_bool = [formulate_label(i, benchmark_list, True) for i in train['RELQ_RELEVANCE2ORGQ']]

# (f) Term frequency
tr_org_q_vecs = np.load('savings/taskB/tr_org_q_vecs.npy') # size = (267, 4827)
tr_rel_q_vecs = np.load('savings/taskB/tr_rel_q_vecs.npy') # size = (2669, 4827)

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

# (b) Similarities - an original query with its relevant queries
def sim_q_djs(qs, rel_qs, K1, b):
    qs_sim_q_djs = []
    for idx in range(len(qs)):
        loc = np.where(train['OrgQBody'] == tr_org_queries[idx])[0]
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

K1 = [0.1, 0.3, 0.5] # smooth factor
b = [0.3, 0.5, 0.7] # b will be set as a value closed to 1 if you want to put emphasis on the length of a document
threshold = [0.75, 0.85, 0.95] # threshold that those transformed similarities that are larger than it will be determined as relevant comments

hyper_param = np.vstack((np.vstack((np.repeat(K1, 9),
                                    np.tile(np.repeat(b, 3), 3))),
                         np.tile(threshold, 9))).T
"""Hyper-parameters combinations

        K1     b  threshold
array([[0.1 , 0.3 , 0.75],
       [0.1 , 0.3 , 0.85],
       [0.1 , 0.3 , 0.95],
       [0.1 , 0.5 , 0.75],
       [0.1 , 0.5 , 0.85],
       [0.1 , 0.5 , 0.95],
       [0.1 , 0.7 , 0.75],
       [0.1 , 0.7 , 0.85],
       [0.1 , 0.7 , 0.95],
       [0.3 , 0.3 , 0.75],
       [0.3 , 0.3 , 0.85],
       [0.3 , 0.3 , 0.95],
       [0.3 , 0.5 , 0.75],
       [0.3 , 0.5 , 0.85],
       [0.3 , 0.5 , 0.95],
       [0.3 , 0.7 , 0.75],
       [0.3 , 0.7 , 0.85],
       [0.3 , 0.7 , 0.95],
       [0.5 , 0.3 , 0.75],
       [0.5 , 0.3 , 0.85],
       [0.5 , 0.3 , 0.95],
       [0.5 , 0.5 , 0.75],
       [0.5 , 0.5 , 0.85],
       [0.5 , 0.5 , 0.95],
       [0.5 , 0.7 , 0.75],
       [0.5 , 0.7 , 0.85],
       [0.5 , 0.7 , 0.95]])
"""
train_accs = []
for param in hyper_param:
    rel = relevance(tr_org_q_vecs, tr_rel_q_vecs, param[0], param[1], param[2])
    train_acc = accuracy(rel, label)
    train_accs.append(train_acc)
    
"""Training accuracies based on different combinations of hyper-parameters

[0.5593855376545522,
 0.5642562757587112,
 0.5687523417010116,
 0.5593855376545522,
 0.5638816035968528,
 0.5676283252154365,
 0.5575121768452604,
 0.5642562757587112,
 0.5672536530535781,
 0.5552641438741102,
 0.5620082427875609,
 0.5668789808917197,
 0.5578868490071188,
 0.5635069314349944,
 0.566129636568003,
 0.5586361933308355,
 0.5623829149494193,
 0.5657549644061446,
 0.554140127388535,
 0.5635069314349944,
 0.5665043087298614,
 0.5563881603596853,
 0.5635069314349944,
 0.5635069314349944,
 0.5567628325215437,
 0.5638816035968528,
 0.5650056200824278]
"""
highest_train_acc = np.max(train_accs) # 0.56875
best_params = hyper_param[np.argmax(train_accs)] # [0.1 , 0.3 , 0.95]
# np.save('savings/taskB/taskB_best_params.npy', best_params)
