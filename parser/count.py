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
	return line.lower()

def add(vocab,line):
	if(line is None):
		return 
	for word in line.strip().split():
		try:
			vocab[word] += 1
		except KeyError:
			vocab[word] = 1

task = 'taskC'
ID = set()

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
	if(data["ID"] in ID):
		continue
	ID.add(data["ID"])

	add(vocab,remove(data["Subject"]))
	add(vocab,remove(data["Body"]))
	if("Comment" in data):
		for comment in data["Comment"]:
			if(comment["ID"] in ID):
				continue
			ID.add(comment["ID"])

			add(vocab,remove(comment["Text"]))

	elif("RelQuestion" in data):
		for ques in data["RelQuestion"]:
			if(ques["ID"] in ID):
				continue
			ID.add(ques["ID"])

			add(vocab,remove(ques["Subject"]))
			add(vocab,remove(ques["Body"]))
	elif("Thread" in data):
		for thread in data["Thread"]:
			add(vocab,remove(thread["RelQuestion"]["Subject"]))
			add(vocab,remove(thread["RelQuestion"]["Body"]))
			for comment in thread["Comment"]:
				add(vocab,remove(comment["Text"]))

arr = [ (name,vocab[name]) for name in vocab]
arr = sorted(arr,key=lambda x:x[1],reverse=True)

with open('vocab.'+task,'w') as f:
	for data in arr:
		f.write('{0} {1}\n'.format(data[0],data[1]))

