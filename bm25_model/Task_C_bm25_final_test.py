#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task C
Relevant query v.s. Relevant comments
BM25_Final_test
"""
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

warning_ignored = np.seterr(divide = 'ignore', invalid = 'ignore')

# 1. Import testing dataset which is saved when running the file, 'Data_preprocessing.py'
final_test = pd.read_csv('data/csv_files/final_test_C.csv').iloc[:, 1:]

# 2. Preprocessing
# (a) Extract texts of all queries and comments from dataset
final_test_queries = np.unique(final_test['OrgQBody'])
final_test_comments = final_test['RelCText']
final_test_q_c = np.hstack((final_test_queries, final_test_comments)) # merge queries and comments

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

final_test_qs = [tokenize(q) for q in final_test_queries]
final_test_cs = [tokenize(c) for c in final_test_comments]

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

hash_final_test_qs = [qc_ngram(q) for q in final_test_qs]
hash_final_test_cs = [qc_ngram(c) for c in final_test_cs]
hash_vocabulary = np.load('savings/taskC/taskC_hash_vocabulary.npy')

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

# final_test_q_vecs = tf(hash_final_test_qs)
# final_test_c_vecs = tf(hash_final_test_cs)

# np.save('savings/taskC/taskC_final_test_q_vecs.npy', final_test_q_vecs)
# np.save('savings/taskC/taskC_final_test_c_vecs.npy', final_test_c_vecs)

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


# (f) Term frequency
final_test_q_vecs = np.load('savings/taskC/taskC_final_test_q_vecs.npy') # size = (880, 8465)
final_test_c_vecs = np.load('savings/taskC/taskC_final_test_c_vecs.npy') # size = (8800, 8465)


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
"""
def bm25(q, djs, K1, b, N):
    avg_len_djs = np.mean([np.sum(dj) for dj in djs]) # average length of all comments given a query
    
    sim_q_djs = [] # similarities of documents with a given query
    # For each of the documents to be ranked given a query
    for j in range(N):
        ki_idx_in_q_dj = np.unique(np.hstack((np.where(q != 0)[0], np.where(djs[j] != 0)[0]))) # index of terms (trigrams)
        fij = djs[j][ki_idx_in_q_dj] # Frequency(i-th trigram in j-th comment)
        len_dj = np.sum(djs[j]) # length of j-th comment
        
        Bij = ((K1 + 1) * fij) / (K1 * (1 - b + b * len_dj / avg_len_djs + 1e-7) + fij + 1e-7)  # term frequency factor with consideration of the length of the comments
                                                                                                # Add 1e-7 to prevent from value = 0
        nis = [np.sum([1 for dj in djs if dj[ki_idx] != 0]) for ki_idx in ki_idx_in_q_dj] # document frequencies
        
        sim_q_dj = np.sum(Bij * np.log2((N - np.float32(nis) + 0.5) / (np.float32(nis) + 0.5))) 
        # similarity calculated based on both term frequency factor and document frequencies (tf-idf)
        sim_q_djs.append(sim_q_dj)
        
    return sim_q_djs
"""

# (b) Similarities
def sim_q_djs(qs, rel_qs, K1, b):
    qs_sim_q_djs = []
    for idx in range(len(qs)):
        loc = np.where(final_test['OrgQBody'] == final_test_queries[idx])[0]
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

# (d) Find the best K1, b and threshold to determine whether it is more accurate
def relevance(qs, rel_qs, K1, b, threshold):
    qs_sim_q_djs = sim_q_djs(qs, rel_qs, K1, b)
    new_sims = []
    for i in qs_sim_q_djs:
        new_sims.extend(trans_sim(i))
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

# Loading best parameters saved when training
best_K1 = np.load('savings/taskC/taskC_best_params.npy')[0]
best_b = np.load('savings/taskC/taskC_best_params.npy')[1]
threshold = np.load('savings/taskC/taskC_best_params.npy')[2]

# (d) Ranking by similarities and save outputs
def rank(sim_q_djs):
    sim_order = np.flip(np.argsort(sim_q_djs), 0)
    rank = np.arange(len(sim_q_djs))
    rank[sim_order] = rank + 1
    return rank

def save_txt(file_name, df):
    with open(file_name, 'w') as f:
        for i in range(len(df)):
            row = df.iloc[i, :]
            for j in row:
                f.write(str(j))
                if j != row[len(row)-1]:
                    f.write('\t')               
            f.write('\n')

# 4. Main 
def main():
    # Similarities calculated and transformed to [0., 1.]
    qs_sim_q_djs = sim_q_djs(final_test_q_vecs, final_test_c_vecs, best_K1, best_b)
    similarities = []
    for i in qs_sim_q_djs:
        similarities.extend(trans_sim(i))
    
    # Ranking
    ranks = []
    for sim in qs_sim_q_djs:
        ranks.extend(rank(sim))
       
    # Relevant or not
    rel = relevance(final_test_q_vecs, final_test_c_vecs, best_K1, best_b, threshold)
    label = []
    for i in rel:
        if i == 1:
            label.append('true')
        else:
            label.append('false')
    ranks = pd.Series(ranks)
    score = pd.Series(similarities)
    label = pd.Series(label)
    
    # Formulate as a data frame
    output = pd.concat((final_test['ORGQ_ID'], final_test['RELC_ID'], ranks, score, label), 1) 
      
    # Save as .txt
    save_txt('savings/taskC/output/taskC_BM25_output.txt', output)


if __name__ == '__main__':
    main()

# =============================================================================
# Evaluation
#
# ******************************
# *** Classification results ***
# ******************************
# 
# Acc = 0.9584
# P   = 0.0455
# R   = 0.0244
# F1  = 0.0317
# 
# 
# ********************************
# *** Detailed ranking results ***
# ********************************
# 
# IR  -- Score for the output of the IR system (baseline).
# SYS -- Score for the output of the tested system.
# 
#            IR   SYS
# MAP   : 0.0918 0.0719
# AvgRec: 0.2172 0.1438
# MRR   :  10.11   7.99
# =============================================================================
