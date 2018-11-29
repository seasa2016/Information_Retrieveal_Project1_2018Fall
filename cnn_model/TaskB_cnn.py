#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task B
Original query v.s. Relevant query
CNN
"""
import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

import torch as t
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.utils.data
from torch import optim
import time

warning_ignored = np.seterr(all = 'ignore', divide = 'ignore', invalid = 'ignore')

# 1. Dataset
# Train
train_A = pd.read_csv('data/csv_files/train_A.csv', encoding = 'utf-8').iloc[:, 1:]
train_B = pd.read_csv('data/csv_files/train_B.csv', encoding = 'utf-8').iloc[:, 1:]
train_C = pd.read_csv('data/csv_files/train_C.csv', encoding = 'utf-8').iloc[:, 1:]


#train_A.columns.values
#train_B.columns.values
#train_C.columns.values

train_data = pd.merge(train_A, train_B, on = ['RELQ_ID', 'RelQBody'])

#train_data.columns.values

# Label
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

#qc_benchmark_list = np.unique(train_data['RELC_RELEVANCE2RELQ'])[1] # relevant
#qc_label = [formulate_label(i, qc_benchmark_list, False) for i in train_data['RELC_RELEVANCE2RELQ']]
#qc_label_bool = [formulate_label(i, qc_benchmark_list, True) for i in train_data['RELC_RELEVANCE2RELQ']]

qq_benchmark_list = np.unique(train_data['RELQ_RELEVANCE2ORGQ'])[1:] # relevant
qq_label = [formulate_label(i, qq_benchmark_list, False) for i in train_data['RELQ_RELEVANCE2ORGQ']]
qq_label_bool = [formulate_label(i, qq_benchmark_list, True) for i in train_data['RELQ_RELEVANCE2ORGQ']]

# Validation
val_A = pd.read_csv('data/csv_files/val_A.csv', encoding = 'utf-8').iloc[:, 1:]
val_B = pd.read_csv('data/csv_files/val_B.csv', encoding = 'utf-8').iloc[:, 1:]
val_data = pd.merge(val_A, val_B, on = ['RELQ_ID', 'RelQBody'])
val_qc_benchmark_list = np.unique(val_data['RELC_RELEVANCE2RELQ'])[1] # relevant
val_qc_label = [formulate_label(i, val_qc_benchmark_list, False) for i in val_data['RELC_RELEVANCE2RELQ']]

# Test
## Task B
test_data = pd.read_csv('data/csv_files/final_test_B.csv', encoding = 'utf-8').iloc[:, 1:]


# 2. Preprocessing
print('Preprocessing...')
# (a) Extract texts of all queries and comments from dataset

tr_OrgQBody = train_data['OrgQBody']
tr_RelQBody = train_data['RelQBody']
tr_RelCText = train_data['RelCText']

val_OrgQBody = val_data['OrgQBody']
val_RelQBody = val_data['RelQBody']
val_RelCText = val_data['RelCText']

## Task B
test_OrgQBody = test_data['OrgQBody']
test_RelQBody = test_data['RelQBody']


# (b) Words tokenization
print('Words tokenization...')
def tokenize(alist):
    letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(alist))
    token = re.sub(r'[)!;/.?:-]', ' ', letters_only)
    token = word_tokenize(token.lower())
    return token

# for each query / comment to tokenize
tr_orgQ_tok = [tokenize(q) for q in tr_OrgQBody] 
tr_relQ_tok = [tokenize(q) for q in tr_RelQBody] 
tr_relC_tok = [tokenize(c) for c in tr_RelCText]

val_orgQ_tok = [tokenize(q) for q in val_OrgQBody] 
val_relQ_tok = [tokenize(q) for q in val_RelQBody] 
val_relC_tok = [tokenize(c) for c in val_RelCText]

## Task B
test_OrgQBody_tok = [tokenize(q) for q in test_OrgQBody] 
test_RelQBody_tok = [tokenize(c) for c in test_RelQBody]

# (c). Remove the stop words
print('Remove the stop words...')
# nltk.download('stopwords') # Download it first
stop_words = set(stopwords.words('english')) # 179 stop words provided in the package
tr_orgQ_stop = [[w for w in q if w not in stop_words] for q in tr_orgQ_tok]
tr_relQ_stop = [[w for w in q if w not in stop_words] for q in tr_relQ_tok]
tr_relC_stop = [[w for w in c if w not in stop_words] for c in tr_relC_tok]
tr_text = np.hstack((tr_orgQ_stop, tr_relQ_stop, tr_relC_stop))

val_orgQ_stop = [[w for w in q if w not in stop_words] for q in val_orgQ_tok]
val_relQ_stop = [[w for w in q if w not in stop_words] for q in val_relQ_tok]
val_relC_stop = [[w for w in c if w not in stop_words] for c in val_relC_tok]

## Task B
test_OrgQBody_stop = [[w for w in q if w not in stop_words] for q in test_OrgQBody_tok]
test_RelQBody_stop = [[w for w in q if w not in stop_words] for q in test_RelQBody_tok]



# 3. Word2vec
print('Word2vec...')
model = Word2Vec(tr_text, size = 256, workers = 8, min_count = 0, iter = 10)

def doc2matrix(qs, model): # input: queries or comments
    qs_vec = []
    for q in qs:
        q_vec = np.zeros(256)
        c = 0
        for w in q:
            if w in model:
                q_vec = np.vstack((q_vec, model[w]))
                c += 1
        if c == 0:
            qs_vec.append(q_vec.reshape([1, -1]))
        else:
            qs_vec.append(q_vec[1:])
    return qs_vec # output: sentece matrics given word2vec model


tr_orgQ = doc2matrix(tr_orgQ_stop, model)
tr_relQ = doc2matrix(tr_relQ_stop, model)
tr_relC = doc2matrix(tr_relC_stop, model)

val_orgQ = doc2matrix(val_orgQ_stop, model)
val_relQ = doc2matrix(val_relQ_stop, model)
val_relC = doc2matrix(val_relC_stop, model)

# test_orgQ = doc2matrix(test_orgQ_stop, model)
test_orgQ = doc2matrix(test_OrgQBody_stop, model)
test_relQ = doc2matrix(test_RelQBody_stop, model)


# 4. Concatenating and Padding
print('Padding...')
max_len_setting = 500

taskB_qq = [np.vstack((tr_orgQ[i], tr_relQ[i])) for i in range(train_data.shape[0])] # concatenate queries and comments

taskB_qq_padding = []
for i in taskB_qq:
    if len(i) < max_len_setting:
        taskB_qq_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        taskB_qq_padding.append(i[:max_len_setting, :])


val_taskB_qq = [np.vstack((val_orgQ[i], val_relQ[i])) for i in range(val_data.shape[0])] # concatenate queries and comments

val_taskB_qq_padding = []
for i in val_taskB_qq:
    if len(i) < max_len_setting:
        val_taskB_qq_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        val_taskB_qq_padding.append(i[:max_len_setting, :])


test_taskB_qq = [np.vstack((test_orgQ[i], test_relQ[i])) for i in range(test_data.shape[0])] # concatenate queries and comments

test_taskB_qq_padding = []
for i in test_taskB_qq:
    if len(i) < max_len_setting:
        test_taskB_qq_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        test_taskB_qq_padding.append(i[:max_len_setting, :])
        
        
# 5. Inputs
trainset = [(taskB_qq_padding[i], qq_label[i]) for i in range(len(taskB_qq_padding))]
validset = [(val_taskB_qq_padding[i], val_qc_label[i]) for i in range(len(val_taskB_qq_padding))]

dataset = {'train': trainset, 'valid': validset}

batch_size = 500
dataloader = {x: t.utils.data.DataLoader(dataset[x], batch_size = batch_size, shuffle = False, 
                                         num_workers = 2) for x in ['train', 'valid']}


# 6. Model - CNN
print('Modeling...')
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.Conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = (3, 256)), # in_channels, out_channels, kernel_size, stride = 1, padding = 0
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 16, (300-3+1)298, 1] -> [1, 16, 149, 1]
                nn.BatchNorm2d(16),
                nn.Dropout(0.2),
                
                nn.Conv2d(16, 32, kernel_size = (3, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 32, (149-3+1)147, 1] -> [1, 32, 73, 1]
                nn.BatchNorm2d(32),
                nn.Dropout(0.2),
                
                nn.Conv2d(32, 32, kernel_size = (5, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 32, (73-5+1)69, 1] -> [1, 32, 34, 1]
                nn.BatchNorm2d(32),           
                nn.Dropout(0.2)
                
        )
        
        self.Classify = nn.Sequential(
                nn.Linear(32* 59*1, 96),  
                nn.ReLU(),
                nn.BatchNorm1d(96),
                nn.Dropout(0.3),
                
                nn.Linear(96, 2),
                #nn.BatchNorm1d(2)
        )
        
    def forward(self, inputs):
        conv = self.Conv(inputs)
        flatten = conv.view(-1, 32* 59*1) 
        output = self.Classify(flatten)
        return output

net = cnn()

# 6. Loss function and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)
          # optim.Adam(net.parameters(), lr = 1e-2)


# 7. Training
print('Training...')
# print(t.cuda.is_available()) # Check GPU
Use_gpu = t.cuda.is_available()

if Use_gpu:
    net = net.cuda()

epoch_n = 10

time_open = time.time()


for epoch in range(epoch_n):
    print('Epoch {}/{} [{}]'.format(epoch + 1, epoch_n, time.time()))
    print('-' * 10)
    
    for phase in ['train', 'valid']:
        if phase == 'train':
            print('Training...')
            net.train(True)
        else:
            print('Validating...')
            net.train(False)
            
        loss = 0.
        corrects = 0
            
        for batch, data in enumerate(dataloader[phase]):
            
            x_batch, y_batch = data
            
            x_batch_reshape = x_batch.view([-1, 1, max_len_setting, 256]).float()
            
            if Use_gpu:
                x, y = Variable(x_batch_reshape.cuda()), Variable(y_batch.cuda())
            else:
                x, y = Variable(x_batch_reshape), Variable(y_batch)
            
            
            y_pred = net(x)
            
            _, pred = t.max(y_pred.data, 1)
            
            optimizer.zero_grad()
            
            loss = loss_fun(y_pred, y)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            loss += loss.item()
            corrects += t.sum(pred == y.data)
            
            # avg_loss = loss / (batch + 1)
            # acc = 100 * corrects / ((batch + 1) * batch_size)
            
        epoch_loss = loss / ( len(dataset[phase]) / batch_size )
        epoch_acc = 100 * corrects / len(dataset[phase])
        
        print('{} Loss: {:.4f} Acc: {:.4f}%\n'.format(phase, epoch_loss, epoch_acc))

time_end = time.time() - time_open
print(time_end)
print('Training Finished')

#t.save(net, 'savings_torch/cnn_model_v2.pt')
t.save(net.state_dict(), 'torch_savings/cnn_taskB.pt')


# 8. Testing
x_test_reshape = np.array(test_taskB_qq_padding).reshape([-1, 1, max_len_setting, 256])

test_dataloader = torch.utils.data.DataLoader(x_test_reshape, batch_size = 100, shuffle = False, 
                                              num_workers = 2)

Use_gpu = t.cuda.is_available()
        
print('Loading model...')
if Use_gpu:
    device = t.device("cuda")
    cnn_model = net
    cnn_model.eval()
    cnn_model.load_state_dict(t.load('torch_savings/cnn_taskB.pt'))
    cnn_model.to(device)
    cnn_model.eval()
else:
    device = t.device("cpu")
    cnn_model = net
    cnn_model.eval()
    cnn_model.load_state_dict(t.load(t.load('torch_savings/cnn_taskB.pt')))
    cnn_model.eval()

print('\nInference...')
test_pred = []
test_similarities = []
for _, x_test_batch in enumerate(test_dataloader):   
    if Use_gpu:
        x_test_tensor = Variable(x_test_batch.cuda())
        test_outputs = cnn_model(x_test_tensor.float())
        test_similarities.extend(t.Tensor.cpu(test_outputs.detach()).numpy().T[1])
        test_pred.extend(t.Tensor.cpu(t.max(test_outputs, 1)[1]).numpy())
    else:
        x_test_tensor = Variable(x_test_batch)
        test_outputs = cnn_model(x_test_tensor)
        test_similarities.extend(test_outputs.data[1].numpy())
        test_pred.extend(t.max(test_outputs, 1)[1].numpy())

# 9. Save file
def save_txt(file_name, df):
    with open(file_name, 'w') as f:
        for i in range(len(df)):
            row = df.iloc[i, :]
            for j in row:
                f.write(str(j))
                if j != row[len(row)-1]:
                    f.write('\t')               
            f.write('\n')
        
def main():
    for i, pred in enumerate(test_pred):
        if pred == 1:
            test_pred[i] = 'true'
        else:
            test_pred[i] = 'false'
            
    ranks = pd.Series(np.arange(len(test_pred)))
    output = pd.concat((test_data['ORGQ_ID'], test_data['RELQ_ID'], ranks, 
                        pd.Series(test_similarities), pd.Series(test_pred)), 1) 
    
    save_txt('torch_savings/taskB_v2/output/taskB_cnn_output_v2.txt', output)
    print('Testing Fininshed')


if __name__ == '__main__':
    main()


# =============================================================================
# Evaluation
#
# ******************************
# *** Classification results ***
# ******************************
# 
# Acc = 0.5761
# P   = 0.2017
# R   = 0.4356
# F1  = 0.2757
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
# MAP   : 0.4185 0.3474
# AvgRec: 0.7759 0.6596
# MRR   :  46.42  39.13
# =============================================================================
