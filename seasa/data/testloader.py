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
#import parser

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
				try:
					arr.append(self.vocab[word])
				except:
					arr.append(0)
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

		dataloader = parser.parser(file_name,task,'test')

		for data in dataloader.iterator():
			temp = {}

			temp['query_ID'] = data["ID"][:-5]
			temp['query'] = remove(data['Subject']) + remove(data['Body'])
			temp['query_len'] = len(temp['query'])
			if(temp['query_len'] == 0):
				temp['query'].append(0)
				temp['query_len'] += 1

		
			if(task == 'taskA'):
				if("Comment" in data):
					for comment in data["Comment"]:
						text = remove(comment["Text"])
						
						temp['answer_ID'] = comment["ID"][:-5]
						
						temp['answer'] = text
						temp['answer_len'] = len(text)
						if(temp['answer_len'] == 0):
							temp['answer'].append(0)
							temp['answer_len'] += 1
		
						self.data.append(temp.copy())

						
			elif(task == 'taskB'):				
				if("RelQuestion" in data):
					for ques in data["RelQuestion"]:
						text = remove(ques["Subject"]) + remove(ques['Body'])
						
						temp['answer_ID'] = ques["ID"]

						temp['answer'] = text
						temp['answer_len'] = len(text)
						if(temp['answer_len'] == 0):
							temp['answer'].append(0)
							temp['answer_len'] += 1
							
						self.data.append(temp.copy())
			elif(task == 'taskC'):				
				for thread in data["Thread"]:
					for comment in thread["Comment"]:
						text = remove(comment["Text"])
					
						temp['answer_ID'] = comment["ID"]
					
						temp['answer'] = text
						temp['answer_len'] = len(text)
						if(temp['answer_len'] == 0):
							temp['answer'].append(0)
							temp['answer_len'] += 1
		
						self.data.append(temp.copy())
			else:
				raise ValueError('qq')
					
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]

		if self.transform:
			sample = self.transform(sample)
		return sample

class ToTensor(object):
	def __call__(self,sample):
		for name in ['query','answer','query_len','answer_len']:
			sample[name] = torch.tensor(sample[name],dtype=torch.long)
			
		return sample

def collate_fn(data):
	"""
	parsing the data list into batch tensor
	"""
	output = dict()

	for name in ['answer_ID','query_ID']:
		output[name] = [ _[name] for _ in data]


	for name in ['query_len','answer_len']:
		temp = [ _[name] for _ in data]	 
		output[name] = torch.stack(temp, dim=0) 
	
	#deal with source and target
	for name in ['answer','query']:
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
	dataset = itemDataset( file_name='./../../data/test_data/SemEval2017-task3-English-test-input.xml',vocab='./vocab',task='taskC',transform=transforms.Compose([ToTensor()]))

	dataloader = DataLoader(dataset, batch_size=2,shuffle=True, num_workers=1,collate_fn=collate_fn)
	for i,data in enumerate(dataloader):
		if(i==0):
			print(data)
	print('finish')
