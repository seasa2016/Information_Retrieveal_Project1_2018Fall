from time import gmtime, strftime
import os
import argparse

from data.dataloader import itemDataset,collate_fn,ToTensor
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

import torch
import torch.optim as optim
import torch.nn as nn

from model.simple_rnn import simple_rnn



def get_data(batch_size):
	"""
	[
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-dev.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-test.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml'
	]
	"""
	train_file = [
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-dev.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml',
	'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml'
	]
	test_file = [
		'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-test.xml'
	]

	train_dataset = itemDataset( file_name=train_file,vocab='./data/vocab',
                                transform=transforms.Compose([ToTensor()]))
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=1,collate_fn=collate_fn)

	valid_dataset = itemDataset( file_name=test_file,vocab='./data/vocab',
								transform=transforms.Compose([ToTensor()]))
	valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=False, num_workers=1,collate_fn=collate_fn)
    
	dataloader = {}
	dataloader['train'] = train_dataloader
	dataloader['valid'] = valid_dataloader

	length = {}
	length['train'] = len(train_dataset)
	length['valid'] = len(valid_dataset)

	return dataloader,length

def convert(data,device):
	for name in data:
		data[name] = data[name].to(device)
	return data

def train(args):
	print("check device")
	if(torch.cuda.is_available() and args.gpu):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader,length = get_data(args.batch_size)
	print("setting model")
	model = simple_rnn(args)
	model = model.to(device=device)

	print(model)
	optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
	criterion = nn.BCEWithLogitsLoss(reduction='sum')
	
	loss_best = 100000000
	print("start training")
	for now in range(args.epoch):
		print(now)
		
		Loss = {'class':0,'rank':0}
		Count = {'class':0,'rank':0}

		model.train()
		for i,data in enumerate(dataloader['train']):
			model.zero_grad()

			#first convert the data into cuda
			data = convert(data,device)

			#deal with the classfication part
			out_left = model.encoder(data['query'],data['query_len'],data['left'],data['left_len'])
			out = model.decode(out_left)
			pred = out>0.5
			Count['class'] += ( data['left_type']==pred ).sum()
			loss = criterion(out,data['left_type']) 
			loss.backward()
			Loss['class'] += loss.detach().cpu().item()

			out_right = model.encoder(data['query'],data['query_len'],data['right'],data['right_len'])
			out = model.decode(out_right)
			pred = out>0.5
			Count['class'] += ( data['right_type']==pred ).sum()
			loss = criterion(out,data['right_type']) 
			loss.backward()
			Loss['class'] += loss.detach().cpu().item()


			#deal with the ranking part
			out = model.rank(out_left,out_right)
			pred = out>0.5
			Count['rank'] +=  pred.sum()
			loss = criterion(out,torch.ones(data['right_type'].shape[0])) 
			Loss['rank'] += loss.detach().cpu().item()
			loss.backward()
			
			optimizer.step()

		if(now%args.print_freq==0):
			print('*'*10)
			print('training loss(class):{0} loss(rank):{1} acc:{2}/{3} {4}/{6}'.format(
							Loss['class'],Loss['rank'],Count['class'],length['train']*2,Count['rank'],length['train']))
		


		Loss = {'class':0,'rank':0}
		Count = {'class':0,'rank':0}

		model.eval()
		for i,data in enumerate(dataloader['valid']):
			with torch.no_grad():
				#first convert the data into cuda
				data = convert(data,device)

				#deal with the classfication part
				out_left = model.encoder(data['query'],data['query_len'],data['left'],data['left_len'])
				out = model.decode(out_left)
				pred = out>0.5
				Count['class'] += ( data['left_type']==pred ).sum()
				loss = criterion(out,data['left_type']) 
				Loss['class'] += loss.detach().cpu().item()

				out_right = model.encoder(data['query'],data['query_len'],data['right'],data['right_len'])
				out = model.decode(out_right)
				pred = out>0.5
				Count['class'] += ( data['right_type']==pred ).sum()
				loss = criterion(out,data['right_type']) 
				Loss['class'] += loss.detach().cpu().item()


				#deal with the ranking part
				out = model.rank(out_left,out_right)
				pred = out>0.5
				Count['rank'] +=  pred.sum()
				loss = criterion(out,torch.ones(data['right_type'].shape[0])) 
				Loss['rank'] += loss.detach().cpu().item()

		if(now%args.print_freq==0):
			print('*'*10)
			print('testing loss(class):{0} loss(rank):{1} acc:{2}/{3} {4}/{6}'.format(
							Loss['class'],Loss['rank'],Count['class'],length['train']*2,Count['rank'],length['train']))
		
		
		torch.save(model.state_dict(), './saved_models/{0}/step_{1}.pkl'.format(args.model_time,now))

		if(Loss['class']<loss_best):
			torch.save(model.state_dict(), './saved_models/{0}/best.pkl'.format(args.model_time))
			loss_best = Loss['class']

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--model', default=0.001, type=str)
	
    parser.add_argument('--print_freq', default=600, type=int)
    parser.add_argument('--word-dim', default=128, type=int)

    args = parser.parse_args()

    setattr(args, 'word_vocab_size', 49522+1)
    setattr(args, 'class_size',1)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    if not os.path.exists('./saved_models/{0}'.format(args.model_time)):
        os.makedirs('./saved_models/{0}'.format(args.model_time))
    
    print('training start!')
    train(args)
    print('training finished!')
	


if(__name__ == '__main__'):
	main()

