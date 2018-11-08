"""
this file is for generate the training data for the bimpm
here we might have something to check that for the relevence of the data
i will first use taskA and taskB on it
"""

import re
import string
from parser.parser import parser


#for the bimpm we limit the length to be less than 200
def remove(line):
	if(line is None):
		return '' 
	line = re.sub('['+string.punctuation+']', ' ', line)
	line = re.sub('  ', ' ', line)

	l = len(line.strip().split())
	if(l>200):
		return None
	return line.lower()

task = 'taskA'
ID = set()

vocab = {}
datalist = {'train':[],'dev':[],'test':[]}
datalist['train'] = [
		'./data/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
		'./data/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
		'./data/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
		'./data/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml',
		'./data/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml'
		]
datalist['dev'] = [
		'./data/training_data/SemEval2016-Task3-CQA-QL-dev.xml'
		]
datalist['test'] = [
		'./data/training_data/SemEval2016-Task3-CQA-QL-test.xml'
		]



		
for dtype in ['train','dev','test']:
	dataloader = parser(datalist[dtype],task)
	
	f = open("./BIMPM-pytorch/data/semeval/{0}_{1}.in".format(task,dtype),'w')
	idx = 0

	for data in dataloader.iterator():
		title = remove(data['Subject']) + ' ' + remove(data['Body'])

		if("Comment" in data):
			for comment in data["Comment"]:
				text = remove(remove(comment["Text"]))
				if('RELEVANCE2RELQ' in comment):
					rel = comment["RELEVANCE2RELQ"]
				elif('RELEVANCE2ORGQ' in comment):
					rel = comment["RELEVANCE2ORGQ"]
				if(rel=='Good'):
					rel = 1
				elif(rel=='Bad'):
					rel = 0
				elif(rel=='PotentiallyUseful'):
					rel = 0

				if(text == None):
					continue

				f.write('{0}\t{1}\t{2}\t{3}\n'.format(rel,title,text,idx))
				idx += 1

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
				idx += 1
		"""	
		elif("Thread" in data):
			for thread in data["Thread"]:
				add(vocab,remove(thread["RelQuestion"]["Subject"]))
				add(vocab,remove(thread["RelQuestion"]["Body"]))
				for comment in thread["Comment"]:
					add(vocab,remove(comment["Text"]))1
		"""

