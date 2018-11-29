import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# File paths
file_dir = '/home/b03901008/IR/Information_retrieval_2018/'
EMBEDDING_FILE = file_dir + 'GoogleNews-vectors-negative300.bin'
MODEL_SAVING_DIR = file_dir

# Load training and test set
def load_data(task):   # e.g. task='A'
    train_df = pd.read_csv(file_dir + 'task_'+str(task)+'_train.csv', index_col=0)
    dev_df = pd.read_csv(file_dir + 'task_'+str(task)+'_test.csv', index_col=0)
    test_2017 = pd.read_csv(file_dir + 'task_'+str(task)+'_2017.csv', index_col=0)
    
    return train_df, dev_df, test_2017

### Choose the desired task ('A', 'B', 'C')
task = 'A'
#task = 'B
#task = 'C'
train_df, dev_df, test_2017 = load_data(task)

if task=='A':
    questions_cols = ['RelCText', 'RelQBody']
elif task=='B':
    questions_cols = ['OrgQ', 'RelQ']
elif task=='C':
    questions_cols = ['OrgQ', 'RelCText']
else:
    questions_cols = []
    print ('Wrong task option!')

    
# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Iterate through sentences, construct embedding mapping
for dataset in [train_df, dev_df, test_2017]:
    for index, row in dataset.iterrows():
        for question in questions_cols:
            # map question to index
            q2n = []  # q2n -> question numbers representation
            for word in row[question]: 
                if word not in word2vec.vocab:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Represent questions with numbers (replace)
            dataset.set_value(index, question, q2n)
            
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # embedding matrix
embeddings[0] = 0  # padding

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

# Create embedding for left & right inputs
max_seq_length = max(train_df[questions_cols[0]].map(lambda x: len(x)).max(),
                     train_df[questions_cols[1]].map(lambda x: len(x)).max(),
                     dev_df[questions_cols[0]].map(lambda x: len(x)).max(),
                     dev_df[questions_cols[1]].map(lambda x: len(x)).max())

X_train = train_df[questions_cols]
Y_train = np.array(train_df['is_relevant'])
X_validation = dev_df[questions_cols]
Y_validation = np.array(dev_df['is_relevant'])

# Split to dicts (left & right)
X_train = {'left': X_train[questions_cols[0]], 'right': X_train[questions_cols[1]]}
X_validation = {'left': X_validation[questions_cols[0]], 'right': X_validation[questions_cols[1]]}
X_test = {'left': test_2017[questions_cols[0]], 'right': test_2017[questions_cols[0]]}

# Zero padding
for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Check left & right size
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)



### Training

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 30
#n_epoch = 25

def exponent_neg_manhattan_distance(left, right):
    ''' The similarity of LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# Input and Embedding Layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Siamese network: shared LSTM for left & right sides
shared_lstm = LSTM(n_hidden)
left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Model & distance calculation of LSTM outputs (between left & right)
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
malstm = Model([left_input, right_input], [malstm_distance])

# Optimizer: Adadelta
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
#malstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Training
training_start_time = time()
malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


# Save Model
json_string = malstm.to_json()

import json
with open('task_'+str(task)+'_json.txt', 'w') as outfile:  
    json.dump(json_string, outfile)
    
print (malstm.summary())



### Testing

def output_result(test_2017, X_test, task):
    df = pd.DataFrame()
    if task=='B':
        df[0] = test_2017['ORGQ_ID']
        df[1] = test_2017['RELQ_ID']
    else:
        df[0] = test_2017['RELQ_ID']
        df[1] = test_2017['RELC_ID']
    df[2] = [1] * len(test_2017)
    preds = malstm.predict([X_test['left'], X_test['right']], verbose=0)
    df[3] = preds
    df[4] = np.where(preds>0.5, 'true', 'false')
    
    return df
output_pred = output_result(test_2017, X_test, task)
output_pred.to_csv('predictions_task'+str(task)+'.pred', sep='\t', header=False, index=False)
