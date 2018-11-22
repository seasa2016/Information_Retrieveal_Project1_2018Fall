"""
this file is for counting the number of word in the dataset


"""

import re
import string
from parser import parser
import sys
import matplotlib.pyplot as plt

def remove(line):
	if(line is None):
		return ''
	line = re.sub('['+string.punctuation+']', ' ', line)
	for i in range(5,1,-1):
		line = re.sub(' '*i, ' ', line)
	
	return line.lower().strip().split()

def add(vocab,line):
	if(line is None):
		return None
	for word in line:
		try:
			vocab[word] += 1
		except KeyError:
			vocab[word] = 1

def compare(stat,num,dtype):
	try:
		if(num and num > stat[dtype]['max']):
			stat[dtype]['max'] = num
		if(num and num < stat[dtype]['min']):
			stat[dtype]['min'] = num
	except KeyError:
		stat[dtype]['min'] = num
		stat[dtype]['max'] = num

def count(stat,num,dtype):
	try:
		stat[dtype][num] += 1
	except KeyError:
		stat[dtype][num] = 1

task = sys.argv[1]
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

#stat = {"Subject":{},"Body":{},"Text":{}}

for data in dataloader.iterator():
	if(data["ID"] in ID):
		continue
	ID.add(data["ID"])

	subject = remove(data["Subject"])
	add(vocab,subject)
	
	body = remove(data["Body"])
	add(vocab,body)

	if("Comment" in data):
		for comment in data["Comment"]:
			if(comment["ID"] in ID):
				continue
			ID.add(comment["ID"])

			text = remove(comment["Text"])
			add(vocab,text)

	elif("RelQuestion" in data):
		for ques in data["RelQuestion"]:
			if(ques["ID"] in ID):
				continue
			ID.add(ques["ID"])
			
			num = len(remove(ques["Subject"]))
			count(stat,num,"Subject")
			num = len(remove(ques["Body"]))
			count(stat,num,"Body")

	elif("Thread" in data):
		for thread in data["Thread"]:
			num = len(remove(thread["RelQuestion"]["Subject"]))
			count(stat,num,"Subject")
			num = len(remove(thread["RelQuestion"]["Body"]))
			count(stat,num,"Body")

			for comment in thread["Comment"]:
				num = len(remove(comment["Text"]))
				count(stat,num,"Text")


arr = [(name,vocab[name]) for name in vocab]
with open('vocab','w') as f:
	for word in arr:
		f.write('{0} {1}\n'.format(word[0],word[1]))

