from time import gmtime, strftime
import os
import argparse

from data.testloader import itemDataset,collate_fn,ToTensor
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

import torch
import torch.optim as optim
import torch.nn as nn

from model.simple_rnn import simple_rnn
from model.qa_lstm import qa_lstm

torch.set_printoptions(threshold=1000)

def get_data(batch_size):
	test_file = [
		'./data/semeval/training_data/SemEval2016-Task3-CQA-QL-test.xml'
	]

	test_dataset = itemDataset( file_name=test_file,vocab='./data/vocab',
								transform=transforms.Compose([ToTensor()]))
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=16,collate_fn=collate_fn)
    
	length = len(test_dataloader)
	
	return test_dataloader,length

def convert(data,device):
	for name in data:
		data[name] = data[name].to(device)
	return data

def train(args):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader,length = get_data(args.batch_size)
	print(length)
	print("setting model")

	if(args.model == 'simple_rnn'):
		model = simple_rnn(args)
	elif(args.model == 'qa_lstm'):
		model = qa_lstm(args)
	elif(args.model == 'bimpm'):
		model = bimpm(args)

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
		temp_Loss = {'class':0,'rank':0}
		temp_Count = {'class':0,'rank':0}

		model.train()
		for i,data in enumerate(dataloader['train']):
			#print(i)
			model.zero_grad()

			#first convert the data into cuda
			data = convert(data,device)

			#deal with the classfication part
			out_left = model.encoder(data['query'],data['query_len'],data['left'],data['left_len'])
			out_left = model.decoder(out_left)
			pred = (out_left.sigmoid()>0.5).int()
			
			temp_Count['class'] += ( data['left_type'].int()==pred ).sum()
			Count['class'] += ( data['left_type'].int()==pred ).sum()
			
			loss = criterion(out_left,data['left_type']) 
			loss.backward(retain_graph=True)
			
			temp_Loss['class'] += loss.detach().cpu().item()
			Loss['class'] += loss.detach().cpu().item()

			out_right = model.encoder(data['query'],data['query_len'],data['right'],data['right_len'])
			out_right = model.decoder(out_right)
			pred = (out_right.sigmoid()>0.5).int()
			temp_Count['class'] += ( data['right_type'].int()==pred ).sum()
			Count['class'] += ( data['right_type'].int()==pred ).sum()
			
			loss = criterion(out_right,data['right_type']) 
			loss.backward(retain_graph=True)
			
			temp_Loss['class'] += loss.detach().cpu().item()
			Loss['class'] += loss.detach().cpu().item()

			#deal with the ranking part
			out = out_left-out_right
			pred = (out.gt(0).int())==data['total_type'].int()

			temp_Count['rank'] +=  pred.sum()
			Count['rank'] +=  pred.sum()
			
			#loss = -(out*data['total_type']).sum()
			loss = criterion(out,data['total_type']) 
			
			temp_Loss['rank'] = loss.detach().cpu().item()
			Loss['rank'] += loss.detach().cpu().item()
			
			loss.backward()
			
			optimizer.step()
			if(i%20==0):
				#print('out',out_right.sigmoid().view(-1))
				#print('label',data['right_type'].view(-1))
				print(i,' training loss(class):{0} loss(rank):{1} acc:{2}/{3} {4}/{5}'.format(temp_Loss['class'],temp_Loss['rank'],temp_Count['class'],args.batch_size*40,temp_Count['rank'],args.batch_size*20))

				temp_Loss = {'class':0,'rank':0}
				temp_Count = {'class':0,'rank':0}
		
		if(now%args.print_freq==0):
			print('*'*10)
			print('training loss(class):{0} loss(rank):{1} acc:{2}/{3} {4}/{5}'.format(
							Loss['class']/length['train']/2,Loss['rank']/length['train'],Count['class'],length['train']*2,Count['rank'],length['train']))
		


		Loss = {'class':0,'rank':0}
		Count = {'class':0,'rank':0}

		model.eval()
		for i,data in enumerate(dataloader['valid']):
			#print(i)
			with torch.no_grad():
				#first convert the data into cuda
				data = convert(data,device)

				#deal with the classfication part
				out_left = model.encoder(data['query'],data['query_len'],data['left'],data['left_len'])
				out_left = model.decoder(out_left)
				pred = (out_left.sigmoid()>0.5).int()
				Count['class'] += ( data['left_type'].int()==pred ).sum()
				loss = criterion(out_left,data['left_type']) 
				Loss['class'] += loss.detach().cpu().item()

				out_right = model.encoder(data['query'],data['query_len'],data['right'],data['right_len'])
				out_right = model.decoder(out_right)
				pred = (out_right.sigmoid()>0.5).int()
				Count['class'] += ( data['right_type'].int()==pred ).sum()
				loss = criterion(out_right,data['right_type']) 
				Loss['class'] += loss.detach().cpu().item()


				#deal with the ranking part
				out = out_left-out_right
				
				loss = criterion(out,data['total_type']) 
				Count['rank'] +=  (out.gt(1e-5).int()==data['total_type'].int()).sum()

				Loss['rank'] += loss.detach().cpu().item()

		if(now%args.print_freq==0):
			print('*'*10)
			print(i,' testing loss(class):{0} loss(rank):{1} acc:{2}/{3} {4}/{5}'.format(
							Loss['class']/length['valid']/2,Loss['rank']/length['valid'],Count['class'],length['valid']*2,Count['rank'],length['valid']))
		
		
		torch.save(model.state_dict(), './saved_models/{0}/step_{1}.pkl'.format(args.save,now))

		if(Loss['class']<loss_best):
			torch.save(model.state_dict(), './saved_models/{0}/best.pkl'.format(args.save))
			loss_best = Loss['class']

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', default=1024, type=int)
	parser.add_argument('--dropout', default=0, type=float)
	parser.add_argument('--epoch', default=200, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	
	parser.add_argument('--word_dim', default=64, type=int)
	parser.add_argument('--hidden_dim', default=64, type=int)
	parser.add_argument('--num_layer', default=2, type=int)

	parser.add_argument('--learning_rate', default=0.005, type=float)
	parser.add_argument('--model', default="qa_lstm", type=str)

	parser.add_argument('--print_freq', default=1, type=int)

	parser.add_argument('--save', required=True , type=str)
	
	args = parser.parse_args()

	setattr(args, 'input_size', 49526+1)
	setattr(args,'batch_first',True)
	setattr(args, 'class_size',1)

	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')

	if not os.path.exists('./saved_models/{0}'.format(args.save)):
		os.makedirs('./saved_models/{0}'.format(args.save))

	print('training start!')
	train(args)
	print('training finished!')
	


if(__name__ == '__main__'):
	main()

