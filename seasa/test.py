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

def get_data(batch_size,task):
	test_file = [
		'./data/semeval/test_data/SemEval2017-task3-English-test-input.xml'
	]

	test_dataset = itemDataset( file_name=test_file,
							vocab='./data/vocab',task=task,transform=transforms.Compose([ToTensor()]))
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=16,collate_fn=collate_fn)
    
	length = len(test_dataloader)
	
	return test_dataloader,length

def convert(data,device):
	for name in data:
		if(name[-2:]=='ID'):
			continue
		data[name] = data[name].to(device)
	return data

def test(args,model_para):
	print("check device")
	if(torch.cuda.is_available() and args.gpu>=0):
		device = torch.device('cuda')
		print('the device is in cuda')
	else:
		device = torch.device('cpu')
		print('the device is in cpu')

	print("loading data")
	dataloader,length = get_data(args.batch_size,args.task)
	print(length)
	print("setting model")

	if(args.model == 'simple_rnn'):
		model = simple_rnn(args)
	elif(args.model == 'qa_lstm'):
		model = qa_lstm(args)
	elif(args.model == 'bimpm'):
		model = bimpm(args)

	model.load_state_dict(model_para)

	model = model.to(device=device)

	print(model)
	
	print("start testing")
		
	with open('./result/'+args.output,'w') as f:
		model.eval()
		for j,data in enumerate(dataloader):
			with torch.no_grad():
				#first convert the data into cuda
				data = convert(data,device)

				#deal with the classfication part
				out_left = model.encoder(data['query'],data['query_len'],data['answer'],data['answer_len'])
				out_left = model.decoder(out_left).detach().cpu()
				ans = (out_left.sigmoid()>0.5).detach().cpu()
				
				for i in range(out_left.shape[0]):
					f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
						data['query_ID'][i],data['answer_ID'][i],i,out_left[i].item(),'true' if(ans[i].item()==1) else 'false'
					))
			print(j)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--load', required=True , type=str)
	parser.add_argument('--task', required=True , type=str)
	ori_args = parser.parse_args()
	setattr(ori_args, 'output','{0}_{1}'.format(args.model,args.task))


	checkpoint = torch.load(ori_args.load)
	print('testing start!')
	
	test(checkpoint['args'],checkpoint['model'])

	print('testing finished!')
	


if(__name__ == '__main__'):
	main()

