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
	def __init__(self, file_name,vocab,cate,transform=None):
		self.data = []
		
		#first build the vocab
		self.vocab = {}
		self.build_dict(self.vocab,vocab)
		
		self.cate = {}
		self.build_dict(self.cate,cate)
		

		#self.add_data(file_name,'taskA')
		self.add_data(file_name,'taskB')

		self.transform = transform
		
		self.total_len = 0

	def build_dict(self,data,vocab):
		with open(vocab) as f:
			for line in f:
				line = line.strip().split()
				data[' '.join(line[:-1])] = len(data)+1

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
			
			if( len(line) > 200 ):
				return []
			return line
		dataloader = parser.parser(file_name,task)
		with open(task,'w') as f:
			json.dump(dataloader.data,f,indent=4)
		
		cc = 0
		for i,data in enumerate(dataloader.iterator()):
			temp = {}
		

			temp['query'] = remove(data['Subject']) + remove(data['Body'])
			temp['query_len'] = len(temp['query'])

			try:
				temp['query_type'] = np.zeros(len(self.cate))
				for name in data['CATEGORY'].split(','):
					temp['query_type'][self.cate[name]-1] += 1
				idx = temp['query_type'].argmax()
				temp['query_type'] = [idx]
			except KeyError:
				temp['query_type'] = [0]




			if(temp['query_len']==0):
				continue
			
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
				for x,y in [(2,1),(1,0),(0,2)]:
					for left in pre[x]:
						for right in pre[y]:
							temp['left'] = left
							temp['left_len'] = len(left)

							temp['right'] = right
							temp['right_len'] = len(right)
							
							temp['left_type'] = 1 if(x>=1) else 0
							temp['right_type'] = 1 if(y>=1) else 0
							
							temp['total_type'] = 1 if(x>y) else 0
							if(temp['left_len']==0 or temp['right_len']==0):
								continue
							self.data.append(temp.copy())

					
			elif("RelQuestion" in data):
				for ques in data["RelQuestion"]:
					text = remove(ques["Subject"]) + remove(ques['Body'])
					rel = ques["RELEVANCE2ORGQ"]
					
					if(rel=='PerfectMatch'):
						rel = 2
					elif(rel=='Relevant'):
						rel = 1
					elif(rel=='Irrelevant'):
						rel = 0
					pre[rel].append(text)
				#parse these into pair
				#parse these into pair
				for x,y in [(2,1),(1,0),(0,2)]:
					for left in pre[x]:
						for right in pre[y]:
							temp['left'] = left
							temp['left_len'] = len(left)

							temp['right'] = right
							temp['right_len'] = len(right)
							
							temp['left_type'] = 1 if(x>=1) else 0
							temp['right_type'] = 1 if(y>=1) else 0

							temp['total_type'] = 1 if(x>y) else 0
							
							if(temp['left_len']==0 or temp['right_len']==0):
								continue
							
							self.data.append(temp.copy())
					

	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample = self.data[idx]
		
		if self.transform:
			sample = self.transform(sample)
		return sample

class ToTensor(object):
	def __call__(self,sample):
		for name in ['query','left','right','query_len','left_len','right_len','query_type']:
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

	for name in ['query_len','left_len','right_len','left_type','right_type','total_type','query_type']:
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
	dataset = itemDataset( file_name='./semeval/training_data/SemEval2016-Task3-CQA-QL-dev.xml',vocab='./vocab',cate='./cate',
								transform=transforms.Compose([ToTensor()]))
	
	
	dataloader = DataLoader(dataset, batch_size=16,shuffle=True, num_workers=1,collate_fn=collate_fn)
	
	for i,data in enumerate(dataloader):
		if(i==0):
			print(data)
	print('finish')
	
