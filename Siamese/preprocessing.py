import os
import pandas as pd
import numpy as np
import Xml2df

# Set the directory that contains training/testing data, 
# e.g. '.../.../Information_retrieval_2018/training_data'
# e.g. '.../.../Information_retrieval_2018/testing_data'
train_dir = ''
test_dir = ''

# Function - 1: remove punctuations in text, set to lower cases

import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
def remove_punctuation(text):
    try:
        _text = regex.sub('', text)
        _text = str(_text).lower()
    except :   # None  UnboundLocalError
        print (text)
        _text = text
    
    return str(_text)
    
## For Task A
def df_processing_A(xml_list):
    train_A = Xml2df.xml2df(xml_list, 'task_A').concat_dfs()
    train_A.dropna(subset=['RelQBody'], inplace=True)
    train_A.reset_index(inplace=True)
      
    RelCText = []
    RelQBody = []
    for i in range(len(train_A)):
        # RelQBody = question subject + body
        RelCText.append(remove_punctuation(train_A['RelCText'][i]))
        RelQBody.append(remove_punctuation(train_A['RelQSubject'][i]) + ' ' + remove_punctuation(train_A['RelQBody'][i]))
        
    rel2relq = np.array(train_A['RELC_RELEVANCE2RELQ'])
    is_relevance = np.where( rel2relq=='Good',1,0)
    
    df = pd.DataFrame()
    df['RELC_ID'] = list(train_A['RELC_ID'])
    df['RELQ_ID'] = list(train_A['RELQ_ID'])
    df['RelCText'] = RelCText
#    df['RelQSubject'] = RelQSubject
    df['RelQBody'] = RelQBody
    df['is_relevant'] = is_relevance
    
    return df

# Train
xml_list = [train_dir+'SemEval2016-Task3-CQA-QL-train-part1.xml', 
           train_dir+'SemEval2016-Task3-CQA-QL-train-part2.xml',
           train_dir+'SemEval2016-Task3-CQA-QL-dev.xml']
eval_A = df_processing_A(xml_list)
eval_A.to_csv('task_A_train.csv')
# Dev
xml_list = [train_dir+'SemEval2016-Task3-CQA-QL-test.xml']
eval_A = df_processing_A(xml_list)
eval_A.to_csv('task_A_dev.csv')
# Test
xml_list = [test_dir+'SemEval2017-task3-English-test-input.xml']
eval_A = df_processing_A(xml_list)
eval_A.to_csv('task_A_2017.csv')


## For Task B
def df_processing_B(xml_list):
    train_A = Xml2df.xml2df(xml_list, 'task_B').concat_dfs()
    train_A.dropna(subset=['OrgQSubject','RelQSubject'], inplace=True)
    train_A.reset_index(inplace=True)   
     
    OrgQ = []
    RelQ = []
    for i in range(len(train_A)):
        # add token 'aaaaa' to separate question subject and body
        # OrgQ / RelQ : question subject + body
        OrgQ.append( remove_punctuation(train_A['OrgQSubject'][i]) + ' aaaaa ' + remove_punctuation(train_A['OrgQBody'][i]))
        RelQ.append( remove_punctuation(train_A['RelQSubject'][i]) + ' aaaaa ' + remove_punctuation(train_A['RelQBody'][i]) )
        
    rel2orgq = np.array(train_A['RELQ_RELEVANCE2ORGQ'])
    is_relevance = np.where( rel2orgq=='Irrelevant',0,1)
    
    df = pd.DataFrame()
    df['ORGQ_ID'] = list(train_A['ORGQ_ID'])
    df['RELQ_ID'] = list(train_A['RELQ_ID'])
    df['OrgQ'] = OrgQ
    df['RelQ'] = RelQ
    df['is_relevant'] = is_relevance
    
    return df

# Train
xml_list = [train_dir+'SemEval2016-Task3-CQA-QL-train-part1.xml', 
           train_dir+'SemEval2016-Task3-CQA-QL-train-part2.xml',
           train_dir+'SemEval2016-Task3-CQA-QL-dev.xml']
eval_B = df_processing_B(xml_list)
eval_B.to_csv('task_B_train.csv')
# Dev
xml_list = [train_dir+'SemEval2016-Task3-CQA-QL-test.xml']
eval_B = df_processing_B(xml_list)
eval_B.to_csv('task_B_dev.csv')
# Test
xml_list = [test_dir+'SemEval2017-task3-English-test-input.xml']
eval_B = df_processing_B(xml_list)
eval_B.to_csv('task_B_2017.csv')


## For Task C
def df_processing_C(purpose):    # e.g. purpose='train', 'dev', 'test'
    eval_A = pd.read_csv('task_A_'+str(purpose)+'.csv', index_col=0)
    eval_B = pd.read_csv('task_B_'+str(purpose)+'.csv', index_col=0)
    eval_C = eval_B.merge(eval_A, on='RELQ_ID', how='left')
    
    rel_ind_B = list(i for i in range(len(eval_C)) if eval_C['is_relevant_x'][i]==1)
    rel_ind_A = list(i for i in range(len(eval_C)) if eval_C['is_relevant_y'][i]==1)
    rel_ind_C = list(set(rel_ind_B).intersection(rel_ind_A))
    
    relevant_C = np.zeros(len(eval_C))
    for i in range(len(rel_ind_C)):
        relevant_C[rel_ind_C[i]] = 1
    
    eval_C['is_relevant'] = relevant_C
    eval_C.drop(columns=['is_relevant_x', 'is_relevant_y'], inplace=True)
    eval_C.to_csv('task_C_'+str(purpose)+'.csv')
    
df_processing_C('train')
df_processing_C('dev')
df_processing_C('2017')
