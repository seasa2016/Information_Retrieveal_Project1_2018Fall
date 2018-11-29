#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR mid-project
Task A
Relevant query v.s. Relevant comments
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

qc_benchmark_list = np.unique(train_data['RELC_RELEVANCE2RELQ'])[1] # relevant
qc_label = [formulate_label(i, qc_benchmark_list, False) for i in train_data['RELC_RELEVANCE2RELQ']]
qc_label_bool = [formulate_label(i, qc_benchmark_list, True) for i in train_data['RELC_RELEVANCE2RELQ']]

#qq_benchmark_list = np.unique(train_data['RELQ_RELEVANCE2ORGQ'])[1:] # relevant
#qq_label = [formulate_label(i, qq_benchmark_list, False) for i in train_data['RELQ_RELEVANCE2ORGQ']]
#qq_label_bool = [formulate_label(i, qq_benchmark_list, True) for i in train_data['RELQ_RELEVANCE2ORGQ']]


# Validation
val_A = pd.read_csv('data/csv_files/val_A.csv', encoding = 'utf-8').iloc[:, 1:]
val_B = pd.read_csv('data/csv_files/val_B.csv', encoding = 'utf-8').iloc[:, 1:]
val_data = pd.merge(val_A, val_B, on = ['RELQ_ID', 'RelQBody'])
val_qc_benchmark_list = np.unique(val_data['RELC_RELEVANCE2RELQ'])[1] # relevant
val_qc_label = [formulate_label(i, val_qc_benchmark_list, False) for i in val_data['RELC_RELEVANCE2RELQ']]

# Test
## Task A
test_A = pd.read_csv('data/csv_files/final_test_A.csv', encoding = 'utf-8').iloc[:, 1:]
test_data = test_A


# 2. Preprocessing
print('Preprocessing...')
# (a) Extract texts of all queries and comments from dataset

tr_OrgQBody = train_data['OrgQBody']
tr_RelQBody = train_data['RelQBody']
tr_RelCText = train_data['RelCText']

val_OrgQBody = val_data['OrgQBody']
val_RelQBody = val_data['RelQBody']
val_RelCText = val_data['RelCText']

## Task A
test_RelQBody = test_data['RelQBody']
test_RelCText = test_data['RelCText']

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

## Task A
# test_orgQ_tok = [tokenize(q) for q in test_OrgQBody] 
test_relQ_tok = [tokenize(q) for q in test_RelQBody] 
test_relC_tok = [tokenize(c) for c in test_RelCText]

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

## Task A
# test_orgQ_stop = [[w for w in q if w not in stop_words] for q in test_orgQ_tok]
test_relQ_stop = [[w for w in q if w not in stop_words] for q in test_relQ_tok]
test_relC_stop = [[w for w in c if w not in stop_words] for c in test_relC_tok]


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
# 
# np.save('savings/tr_orgQ.npy', tr_orgQ)
# np.save('savings/tr_relQ.npy', tr_relQ)
# np.save('savings/tr_relC.npy', tr_relC)

#tr_orgQ = np.load('savings/tr_orgQ.npy')
#tr_relQ = np.load('savings/tr_relQ.npy')
#tr_relC = np.load('savings/tr_relC.npy')


val_orgQ = doc2matrix(val_orgQ_stop, model)
val_relQ = doc2matrix(val_relQ_stop, model)
val_relC = doc2matrix(val_relC_stop, model)

# test_orgQ = doc2matrix(test_orgQ_stop, model)
test_relQ = doc2matrix(test_relQ_stop, model)
test_relC = doc2matrix(test_relC_stop, model)


# 4. Concatenating and Padding
print('Padding...')
max_len_setting = 500

taskA_qc = [np.vstack((tr_relQ[i], tr_relC[i])) for i in range(train_data.shape[0])] # concatenate queries and comments
# max_len = np.max([len(i) for i in taskA_qc]) # max length of concated query-comment texts
# padding with zeros if length of query-comment texts < max. one
# np.save('savings/taskA_qc.npy', taskA_qc)
#taskA_qc = np.load('savings/taskA_qc.npy')

taskA_qc_padding = []
for i in taskA_qc:
    if len(i) < max_len_setting:
        taskA_qc_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        taskA_qc_padding.append(i[:max_len_setting, :])
#np.save('torch_savings/taskA_qc_padding.npy', taskA_qc_padding)

"""
#taskA_qc_padding_01 = []
#for i in taskA_qc[:5000]:
#    if len(i) < max_len_setting:
#        taskA_qc_padding_01.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
#    else:
#        taskA_qc_padding_01.append(i[:max_len_setting, :])
# np.save('savings/taskA_qc_padding_01.npy', taskA_qc_padding_01)
taskA_qc_padding_01 = np.load('savings/taskA_qc_padding_01.npy')
#taskA_qc_padding_02 = []
#for i in taskA_qc[5000:10000]:
#    if len(i) < max_len_setting:
#        taskA_qc_padding_02.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
#    else:
#        taskA_qc_padding_02.append(i[:max_len_setting, :])
# np.save('savings/taskA_qc_padding_02.npy', taskA_qc_padding_02)
taskA_qc_padding_02 = np.load('savings/taskA_qc_padding_02.npy')
#taskA_qc_padding_03 = []
#for i in taskA_qc[10000:15000]:
#    if len(i) < max_len_setting:
#        taskA_qc_padding_03.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
#    else:
#        taskA_qc_padding_03.append(i[:max_len_setting, :])
# np.save('savings/taskA_qc_padding_03.npy', taskA_qc_padding_03)
taskA_qc_padding_03 = np.load('savings/taskA_qc_padding_03.npy')
#taskA_qc_padding_123 = []
#taskA_qc_padding_123.extend(taskA_qc_padding_01)
#taskA_qc_padding_123.extend(taskA_qc_padding_02)
#taskA_qc_padding_123.extend(taskA_qc_padding_03)
#np.save('savings/taskA_qc_padding_123.npy', taskA_qc_padding_123)
taskA_qc_padding_123 = np.load('savings/taskA_qc_padding_123.npy')

#taskA_qc_padding_04 = []
#for i in taskA_qc[15000:]:
#    if len(i) < max_len_setting:
#        taskA_qc_padding_04.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
#    else:
#        taskA_qc_padding_04.append(i[:max_len_setting, :])
#np.save('savings/taskA_qc_padding_04.npy', taskA_qc_padding_04)
taskA_qc_padding_04 = np.load('savings/taskA_qc_padding_04.npy')

taskA_qc_padding = []
taskA_qc_padding.extend(taskA_qc_padding_123)
taskA_qc_padding.extend(taskA_qc_padding_04)

np.save('savings/taskA_qc_padding.npy', taskA_qc_padding)
"""
val_taskA_qc = [np.vstack((val_relQ[i], val_relC[i])) for i in range(val_data.shape[0])] # concatenate queries and comments
# val_max_len = np.max([len(i) for i in val_taskA_qc]) # max length of concated query-comment texts
# padding with zeros if length of query-comment texts < max. one
# np.save('savings/val_taskA_qc.npy', val_taskA_qc)
#val_taskA_qc = np.load('savings/val_taskA_qc.npy')

val_taskA_qc_padding = []
for i in val_taskA_qc:
    if len(i) < max_len_setting:
        val_taskA_qc_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        val_taskA_qc_padding.append(i[:max_len_setting, :])

test_taskA_qc = [np.vstack((test_relQ[i], test_relC[i])) for i in range(test_data.shape[0])] # concatenate queries and comments
# test_max_len = np.max([len(i) for i in test_taskA_qc]) # max length of concated query-comment texts
# padding with zeros if length of query-comment texts < max. one
# np.save('savings/test_taskA_qc.npy', test_taskA_qc)
#test_taskA_qc = np.load('savings/test_taskA_qc.npy')

test_taskA_qc_padding = []
for i in test_taskA_qc:
    if len(i) < max_len_setting:
        test_taskA_qc_padding.append(np.vstack((i, np.zeros([max_len_setting - len(i), 256]))))
    else:
        test_taskA_qc_padding.append(i[:max_len_setting, :])
             
        
# 5. Inputs
trainset = [(taskA_qc_padding[i], qc_label[i]) for i in range(len(taskA_qc_padding))]
validset = [(val_taskA_qc_padding[i], val_qc_label[i]) for i in range(len(val_taskA_qc_padding))]

dataset = {'train': trainset, 'valid': validset}

batch_size = 600
dataloader = {x: t.utils.data.DataLoader(dataset[x], batch_size = batch_size, shuffle = False, 
                                         num_workers = 2) for x in ['train', 'valid']}

        
# 6. Model - CNN
print('Modeling...')


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.Conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = (3, 256)), # in_channels, out_channels, kernel_size, stride = 1, padding = 0
                nn.LeakyReLU(0.05, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 16, (500-3+1)498, 1] -> [1, 16, 249, 1]
                nn.BatchNorm2d(16),
                nn.Dropout(0.3),
                
                nn.Conv2d(16, 32, kernel_size = (3, 1)),
                nn.LeakyReLU(0.05, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 32, (249-3+1)247, 1] -> [1, 32, 123, 1]
                nn.BatchNorm2d(32),
                nn.Dropout(0.3),

                nn.Conv2d(32, 32, kernel_size = (5, 1)),
                nn.LeakyReLU(0.05, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 32, (123-5+1)119, 1] -> [1, 32, 59, 1]
                nn.BatchNorm2d(32),
                nn.Dropout(0.3)
                
#                nn.Conv2d(32, 32, kernel_size = (32, 1)), 
#                nn.LeakyReLU(0.05, inplace = True),
#                nn.MaxPool2d(kernel_size = (2, 1)), # [1, 32, (59-32+1)28, 1] -> [1, 32, 14, 1]
#                nn.BatchNorm2d(32),
#                nn.Dropout(0.2)
                
                
                
        )
        
        self.Classify = nn.Sequential(
                nn.Linear(32* 59*1, 96),  # nn.Linear(32 * 3 * 1, 96)
                nn.LeakyReLU(0.001, inplace = True),
                nn.BatchNorm1d(96),
                nn.Dropout(0.3),
                
                
#                nn.Linear(96, 48),
#                nn.ReLU(),
#                nn.BatchNorm1d(48),
#                nn.Dropout(0.2),
#                
#                
#                nn.Linear(48, 24),
#                nn.ReLU(),
#                nn.BatchNorm1d(24),
#                nn.Dropout(0.2),
#                
#                nn.Linear(24, 12),
#                nn.ReLU(),
#                nn.BatchNorm1d(12),
#                nn.Dropout(0.2),                
#                
#                nn.Linear(12, 2)
                
                nn.Linear(96, 2),
                #nn.BatchNorm1d(2)
        )
        
    def forward(self, inputs):
        conv = self.Conv(inputs)
        flatten = conv.view(-1, 32* 59*1) #(-1, 32 * 3 * 1)
        output = self.Classify(flatten)
        return output

net = cnn()

"""
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # Convolution x 5
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = (5, 256))
        self.conv2 = nn.Conv2d(in_channels = 5, out_channels = 9, kernel_size = (5, 1))
        self.conv3 = nn.Conv2d(in_channels = 9, out_channels = 11, kernel_size = (3, 1))
        self.conv4 = nn.Conv2d(in_channels = 11, out_channels = 9, kernel_size = (3, 1))
        self.conv5 = nn.Conv2d(in_channels = 9, out_channels = 5, kernel_size = (3, 1))
        
        # Fully connected x 3
        self.fc1 = nn.Linear(5 * 7 * 1, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 2)
    
    def forward(self, x): # x = (num_img, channel_size, w_px, h_px) [1, 1, 152, 256]
        # Conv -> Maxpooling -> relu
        fm1 = F.relu(F.max_pool2d(self.conv1(x), (2, 1))) # [1, 5, 28 (300-5+1)296, 1] -> [1, 5, 148, 1]
        fm2 = F.relu(F.max_pool2d(self.conv2(fm1), (2, 1))) # [1, 9, 10 (148-5+1)144, 1] -> [1, 9, 72, 1]
        fm3 = F.relu(F.max_pool2d(self.conv3(fm2), (2, 1))) # [1, 11, 10 (72-3+1)70, 1] -> [1, 11, 35, 1]
        fm4 = F.relu(F.max_pool2d(self.conv4(fm3), (2, 1))) # [1, 9, 10 (35-3+1)33, 1] -> [1, 9, 16, 1]
        fm5 = F.relu(F.max_pool2d(self.conv5(fm4), (2, 1))) # [1, 5, 10 (16-3+1)14, 1] -> [1, 5, 7, 1]
        
        # Flatten
        flat = fm5.view(fm5.size()[0], -1) # [1, 5 * 7 * 1]
        
        # Fully connected
        out1 = F.relu(self.fc1(flat)) # [1, 30]
        out2 = F.relu(self.fc2(out1)) # [1, 10]
        out3 = F.softmax(self.fc3(out2)) # [1, 2]
        
        return out3

net = AlexNet()
#print(net)


 Check if parameters of AlexNet() are correct 

tmp = taskA_qc_padding[0].astype(np.float).reshape([1, 1, 300, 256])
# Inp = Variable(t.rand(1, 1, 300, 256)) # num_imgs, channel, width_px, height_px
Inp = Variable(t.Tensor(tmp))
NN = AlexNet()
NN.forward(Inp).size()
"""

# 6. Loss function and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.8, momentum = 0.9)
          # optim.Adam(net.parameters(), lr = 1e-2)
          
# 7. Training
print('Training...')
# print(t.cuda.is_available()) # Check GPU
Use_gpu = t.cuda.is_available()

if Use_gpu:
    net = net.cuda()

epoch_n = 25

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
t.save(net.state_dict(), 'torch_savings/cnn_taskA_02.pt')

"""
epoch_size = 10
batch_size = 50

train_x = taskA_qc_padding
train_y = qc_label

val_x = val_taskA_qc_padding
val_y = val_qc_label

test_x = test_taskA_qc_padding

for epoch in range(epoch_size):
    losses = 0.
    
    for i in range(int(len(taskA_qc_padding) / batch_size)):
        ## Training data
        # Input data in batches
        train_x_batch_raw = np.array(train_x[i * batch_size : (i + 1) * batch_size]).reshape([batch_size, 1, 1000, 256])
        train_y_batch_raw = np.array(train_y[i * batch_size : (i + 1) * batch_size])
        
        train_x_batch = torch.Tensor(train_x_batch_raw)
        train_y_batch = torch.Tensor(train_y_batch_raw)
        

        # Capsulate in Variable so that gradients can be calculated
        train_x_batch, train_y_batch = Variable(train_x_batch), Variable(train_y_batch)
        
        # Zerolize gradients
        optimizer.zero_grad()
        
        # forward + backward
        outputs = net(train_x_batch)
        loss = criterion(outputs, train_y_batch.long())
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        losses += loss.data
        avg_loss = losses / (i + 1)
        
        # Training Accuracy
        train_pred = net(train_x).data.max(1)[1]
        train_acc = np.round(np.mean(train_pred.numpy() == train_y) * 100, 2)
        
        ## Validation data
        val_x = torch.Tensor(np.array(val_x).reshape([-1, 1, 300, 256]))
        val_pred = net(val_x).data.max(1)[1]
        val_acc = np.round(np.mean(val_pred.numpy() == val_y) * 100, 2)

        
#        if (i + 1) % 20 == 0: # Every 200 batches, print once 
#            print('(Epoch:%0d, Batch:%5d, Loss:%.4f, Train_acc: %.2f%%)' % (epoch + 1, i + 1, avg_loss, train_acc))
    
    print('(Epoch:%0d, Loss:%.4f, Train_acc: %.2f%%, Val_acc: %.2f%%)' % (epoch + 1, avg_loss, train_acc, val_acc))
            
print('Training process is finished.')
"""

# 8. Testing
x_test_reshape = np.array(test_taskA_qc_padding).reshape([-1, 1, max_len_setting, 256])

test_dataloader = torch.utils.data.DataLoader(x_test_reshape, batch_size = 100, shuffle = False, 
                                              num_workers = 2)

Use_gpu = t.cuda.is_available()
        
print('Loading model...')
if Use_gpu:
    device = t.device("cuda")
    cnn_model = net
    cnn_model.eval()
    cnn_model.load_state_dict(t.load('torch_savings/cnn_taskA_02.pt'))
    cnn_model.to(device)
    cnn_model.eval()
else:
    device = t.device("cpu")
    cnn_model = net
    cnn_model.eval()
    cnn_model.load_state_dict(t.load(t.load('torch_savings/cnn_taskA_02.pt')))
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
        #print('x_test_batch:{}'.format(x_test_batch.data))
        #print('test_outputs:{}'.format(test_outputs.data))
        #print('output:{}'.format(t.Tensor.cpu(t.max(test_outputs, 1)[1]).numpy()))
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
    output = pd.concat((test_A['RELQ_ID'], test_A['RELC_ID'], ranks, 
                        pd.Series(test_similarities), pd.Series(test_pred)), 1) 
    
    save_txt('torch_savings/taskA_v2/output2/taskA_cnn_output_v2.txt', output)
    print('Testing Fininshed')


if __name__ == '__main__':
    main()


# =============================================================================
# Evaluation
#
# ******************************
# =============================================================================
# *** Classification results ***
# ******************************
# 
# Acc = 0.6073
# P   = 0.8707
# R   = 0.2837
# F1  = 0.4280
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
# MAP   : 0.7210 0.8267
# AvgRec: 0.7928 0.8884
# MRR   :  84.19  90.32
# =============================================================================


