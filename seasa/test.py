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

	test_dataset = itemDataset( file_name=test_file,
							vocab='./vocab',task='taskA',transform=transforms.Compose([ToTensor()]))
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=16,collate_fn=collate_fn)
    
	length = len(test_dataloader)
	
	return test_dataloader,length

def convert(data,device):
	for name in data:
		if(name[-2:]=='ID'):
			continue
		data[name] = data[name].to(device)
	return data

def test(args):
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

	model.load_state_dict(torch.load(args.load)

	model = model.to(device=device)

	print(model)
	
	print("start training")
		
	with open(args.output,'w') as f:
		model.eval()
		for i,data in enumerate(dataloader):
			#print(i)
			with torch.no_grad():
				#first convert the data into cuda
				data = convert(data,device)

				#deal with the classfication part
				out_left = model.encoder(data['query'],data['query_len'],data['answer'],data['answer_len'])
				out_left = model.decoder(out_left).detach().cpu()
				ans = (out_left.sigmoid()>0.5).detach().cpu()
				
				for i in range(out_left.shape[0]):
					f.write('{0}\t{1}\t{2}\t{3}\t{4}'.format(
						data['answer_ID'][i],data['query_ID'][i],i,out_left[i],ans[i]
					))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', default=1024, type=int)
	parser.add_argument('--dropout', default=0, type=float)
	parser.add_argument('--gpu', default=0, type=int)
	
	parser.add_argument('--word_dim', default=64, type=int)
	parser.add_argument('--hidden_dim', default=64, type=int)
	parser.add_argument('--num_layer', default=2, type=int)

	parser.add_argument('--learning_rate', default=0.005, type=float)
	parser.add_argument('--model', default="qa_lstm", type=str)

	parser.add_argument('--output', required=True , type=str)
	parser.add_argument('--load', required=True , type=str)
	
	args = parser.parse_args()

	setattr(args, 'input_size', 49526+1)
	setattr(args,'batch_first',True)
	setattr(args, 'class_size',1)
	
	print('testing start!')
	test(args)
	print('testing finished!')
	


if(__name__ == '__main__'):
	main()

