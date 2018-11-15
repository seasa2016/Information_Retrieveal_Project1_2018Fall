"""
this is the simply dataloader for the iquestion answering selection.
I would like to implement the  method with classify and ranking
"""


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import pandas as pd
import sys
import parser

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_name,transform=None):
       
		data_load = parser(file_name,'taskA') 
		add_data(parse,'taskA')
		data_load = parser(file_name,'taskB')
		add_data(parse,'taskB')

        self.transform = transform
	def add_data(self,parse,task='taskA'):
		
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
            
