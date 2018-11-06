"""
this file is for counting the number of word in the dataset


"""

import re
import string
from parser import parser

def remove(line):
	if(line is None):
		return 
	line = re.sub('['+string.punctuation+']', ' ', line)
	line = re.sub('  ', ' ', line)
	return line

def add(vocab,line):
	if(line is None):
		return 
	for word in line.strip().split():
		try:
			vocab[word] += 1
		except KeyError:
			vocab[word] = 1

task = 'taskB'


vocab = {}
datalist = [
		'./../data/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
		'./../data/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
		'./../data/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
		'./../data/training_data/SemEval2016-Task3-CQA-QL-dev.xml',
		'./../data/training_data/SemEval2016-Task3-CQA-QL-test.xml',
		'./../data/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml',
		'./../data/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml'
		]
dataloader = parser(datalist,task)


for data in dataloader.iterator():
	
	add(vocab,remove(data["Subject"]))
	add(vocab,remove(data["Body"]))
	if("Comment" in data):
		for comment in data["Comment"]:
			add(vocab,remove(comment["Text"]))

	if("RelQuestion" in data):
		for comment in data["RelQuestion"]:
			add(vocab,remove(comment["Subject"]))
			add(vocab,remove(comment["Body"]))

arr = [ (name,vocab[name]) for name in vocab]
arr = sorted(arr,key=lambda x:x[1],reverse=True)

with open('vocab.'+task,'w') as f:
	for data in arr:
		f.write('{0} {1}\n'.format(data[0],data[1]))

