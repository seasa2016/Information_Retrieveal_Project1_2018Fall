"""
this is the simply dataloader for the iquestion answering selection.
I would like to implement the  method with classify and ranking
"""


from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import pandas as pd
import sys
from .parser import parser
import re
import string

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_name,vocab,transform=None):
        self.data = []
        
        #first build the vocab
        self.build_dict(vocab)

        self.add_data(file_name,'taskA')
        self.add_data(file_name,'taskB')

        self.transform = transform
        
        self.total_len = 0

    def build_dict(self,vocab):
        self.vocab = {}
        with open(vocab) as f:
            for line in f:
                line = line.strip().split()[0]
                self.vocab[line] = len(self.vocab)

    def add_data(self,file_name,task='taskA'):
        def replace(line):
            arr = []
            for word in line.split():
                arr.append(self.vocab(word))
            return arr

        def remove(line):
            if(line is None):
                return '' 
            line = re.sub('['+string.punctuation+']', ' ', line)
            for i in range(5,1,-1):
                line = re.sub(' '*i, ' ', line)
            line = line.lower()
            line = replace(line)
            
            if( len(line) > 200 ):
                return None
            return line
        
        dataloader = parser(file_name,task)

        for data in dataloader.iterator():
            temp = {}
        
            temp['query'] = remove(data['Subject']) + remove(data['Body'])
            temp['query_len'] = len(temp['query'])
            
            pre = {2:[],1:[],0:[]}

            if("Comment" in data):
                for comment in data["Comment"]:
                    text = remove(comment["Text"])
                    
                    if('RELEVANCE2RELQ' in comment):
                        rel = comment["RELEVANCE2RELQ"]
                    elif('RELEVANCE2ORGQ' in comment):
                        rel = comment["RELEVANCE2ORGQ"]
                    if(rel=='Good'):
                        rel = 2
                    elif(rel=='Bad'):
                        rel = 0
                    elif(rel=='PotentiallyUseful'):
                        rel = 1

                    if(text == None):
                        continue

                    pre[rel].append(text)

                #parse these into pair
                for x,y in [(2,1),(1,0),(2,0)]:
                    for left in pre[x]:
                        for right in pre[y]:
                            temp['left'] = left
                            temp['left_len'] = len(left)

                            temp['right'] = right
                            temp['right_len'] = len(right)
                            
                            temp['left_type'] = 1 if(x>=1) else 0
                            temp['right_type'] = 1 if(x>=1) else 0
                            
                            self.data.append(temp)


                    
            elif("RelQuestion" in data):
                for ques in data["RelQuestion"]:
                    text = remove(ques["Subject"]) + ' ' + remove(ques['Body'])
                    rel = ques["RELEVANCE2ORGQ"]
                    
                    if(rel=='PerfectMatch'):
                        rel = 2
                    elif(rel=='Relevant'):
                        rel = 1
                    elif(rel=='Irrelevant'):
                        rel = 0
                    temp['answer'][rel].append(text)
                #parse these into pair
                #parse these into pair
                for x,y in [(2,1),(1,0),(2,0)]:
                    for left in pre[x]:
                        for right in pre[y]:
                            temp['left'] = left
                            temp['left_len'] = len(left)

                            temp['right'] = right
                            temp['right_len'] = len(right)
                            
                            temp['left_type'] = 1 if(x>=1) else 0
                            temp['right_type'] = 1 if(x>=1) else 0
                            
                            self.data.append(temp)
                    

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        for name in sample:
            sample[name] = torch.tensor(sample[name],dtype=torch.long)
        return sample

def collate_fn(data):
    output = dict()
    #deal with source and target
    for t in ['query','left','right']:
        l = data

        for i in range(len(data)):
            if(l-data[i][t].shape[0]):
                data[i][t] =  torch.cat([data[i][t],torch.zeros(l-data[i][t].shape[0],dtype=torch.long)],dim=-1)
    
    
    return output

if(__name__ == '__main__'):
    print('QQQ')
    dataset = itemDataset(file_name='playlist_20181023_train.csv',
                                transform=transforms.Compose([ToTensor()]))
    
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False, num_workers=10,collate_fn=collate_fn)

    for i,data in enumerate(dataloader):
        #if(i==0):
        #    print(data) 
        #break
        print(i)
            
