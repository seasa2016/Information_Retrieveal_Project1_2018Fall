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

from data import parser

import re
import string
import json

#add sos bos
class itemDataset(Dataset):
    def __init__(self, file_name,vocab,task='taskA',transform=None):
        self.data = []
        
        #first build the vocab
        self.build_dict(vocab)

        self.add_data(file_name,task)

        self.transform = transform
        
    def build_dict(self,vocab):
        self.vocab = {}
        with open(vocab) as f:
            for line in f:
                line = line.strip().split()[0]
                self.vocab[line] = len(self.vocab)+1

    def add_data(self,file_name,task='taskA'):
        def replace(line):
            arr = []
            for word in line.split():
                arr.append(self.vocab[word])
            return arr

        def remove(line):
            if(line is None):
                return []
            line = re.sub('['+string.punctuation+']', ' ', line)
            for i in range(5,1,-1):
                line = re.sub(' '*i, ' ', line)
            line = line.lower()
            line = replace(line)
            
            return line

        dataloader = parser.parser(file_name,task)
        with open(task,'w') as f:
            json.dump(dataloader.data,f,indent=4)
        """
        for data in dataloader.iterator():
            temp = {}

            temp['query'] = remove(data['Subject']) + remove(data['Body'])
            temp['query_len'] = len(temp['query'])
            if(temp['query_len']==0):
                continue
			
            if("Comment" in data):
                for comment in data["Comment"]:
                    text = remove(comment["Text"])

                    pre[rel].append(text)
					
            if("RelQuestion" in data):
                for ques in data["RelQuestion"]:
                    text = remove(ques["Subject"]) + remove(ques['Body'])

                    pre[rel].append(text)
        """         
					

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
	def __call__(self,sample):
		for name in ['query','left','right','query_len','left_len','right_len']:
			sample[name] = torch.tensor(sample[name],dtype=torch.long)
			
		for name in ['left_type','right_type','total_type']:
			sample[name] = torch.tensor(sample[name],dtype=torch.float)
			
		return sample

def collate_fn(data):
	"""
	parsing the data list into batch tensor
	['query','left','right']
	['query_len','left_len','right_len']
	['left_type','right_type','total_type']
	"""
	output = dict()

	for name in ['query_len','left_len','right_len','left_type','right_type','total_type']:
		temp = [ _[name] for _ in data]	 
		output[name] = torch.stack(temp, dim=0) 
	
	for name in ['left_type','right_type','total_type']:
		output[name] = output[name].view(-1,1)

	#deal with source and target
	for name in ['right','query','left']:
		length = output['{0}_len'.format(name)]
		l = length.max().item()

		for i in range(len(data)):
			if(l-length[i].item()>0):
				data[i][name] =  torch.cat([data[i][name],torch.zeros(l-length[i].item(),dtype=torch.long)],dim=-1)

		temp = [ _[name] for _ in data]
		output[name] = torch.stack(temp, dim=0).long()
		
	
	return output

if(__name__ == '__main__'):
	print('QQQ')
	dataset = itemDataset( file_name='./semeval/training_data/SemEval2016-Task3-CQA-QL-dev.xml',vocab='./vocab',
								transform=transforms.Compose([ToTensor()]))
	
	
	dataloader = DataLoader(dataset, batch_size=16,shuffle=True, num_workers=1,collate_fn=collate_fn)
	
	for i,data in enumerate(dataloader):
		if(i==0):
			print(data)
	print('finish')
	
