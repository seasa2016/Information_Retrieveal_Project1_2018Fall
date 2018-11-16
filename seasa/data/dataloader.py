"""
this is the simply dataloader for the iquestion answering selection.
I would like to implement the  method with classify and ranking
"""


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import pandas as pd
import sys
from .parser import parser
import re
import string

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_name,transform=None):
       
        self.data = []

        self.add_data(file_name,'taskA')
        self.add_data(file_name,'taskB')

        self.transform = transform
    def add_data(self,file_name,task='taskA'):
        def remove(line):
            if(line is None):
                return '' 
            line = re.sub('['+string.punctuation+']', ' ', line)
            line = re.sub('  ', ' ', line)

            l = len(line.strip().split())
            if(l>200):
                return None
            return line.lower().strip()
        
        dataloader = parser(file_name,task)

        for data in dataloader.iterator():
            temp = {}
            temp['query'] = remove(data['Subject']) + ' ' + remove(data['Body'])
            temp['answer'] = {2:[],1:[],0:[]}

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

                    f.write('{0}\t{1}\t{2}\t{3}\n'.format(rel,title,text,idx))
                    
            elif("RelQuestion" in data):
                for ques in data["RelQuestion"]:
                    text = remove(ques["Subject"]) + ' ' + remove(ques['Body'])
                    rel = ques["RELEVANCE2ORGQ"]
                    
                    if(rel=='PerfectMatch'):
                        rel = 1
                    elif(rel=='Relevant'):
                        rel = 0
                    elif(rel=='Irrelevant'):
                        rel = 0
                    
                    f.write('{0}\t{1}\t{2}\t{3}\n'.format(rel,title,text,idx))
                    

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        return{
            'source':torch.tensor(sample['source'],dtype=torch.long),
            'target':torch.tensor(sample['target'],dtype=torch.long),
            'source_len':torch.tensor(sample['source_len'],dtype=torch.long),
            'target_len':torch.tensor(sample['target_len'],dtype=torch.long),
            'origin':torch.tensor(qq,dtype=torch.long)
            }

def collate_fn(data):
    
    output = dict()
    #deal with source and target
    for t in ['source','target','origin']:
        l = 0
        for i in range(len(data)):
            l = max(l,data[i][t].shape[0])
        if(l == 0):
            continue
        for i in range(len(data)):
            if(l-data[i][t].shape[0]):
                data[i][t] =  torch.cat([data[i][t],torch.zeros(l-data[i][t].shape[0],dtype=torch.long)],dim=-1)
    
    
    for name in [ 'source','target','origin']:
        if(name not in data[0]):
            continue

        arr = [ data[i][name] for i in range(len(data))]
        output[name] = torch.stack(arr,dim=0)
    
    output['source'] = output['source'].transpose(0,1)
    output['source_len'] = torch.cat([ data[i]['source_len'] for i in range(len(data))],dim=0)
    if('target' in output):
        output['target'] = output['target'].transpose(0,1)
        output['target_len'] = torch.cat([ data[i]['target_len'] for i in range(len(data))],dim=0)
    
    return output

if(__name__ == '__main__'):
    print('QQQ')
    dataset = itemDataset(file_name='playlist_20181023_train.csv',
                                transform=transforms.Compose([ToTensor()]))
    
    
    dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=False, num_workers=10,collate_fn=collate_fn)

    for i,data in enumerate(dataloader):
        #if(i==0):
        #    print(data) 
        #break
        print(i)
            
