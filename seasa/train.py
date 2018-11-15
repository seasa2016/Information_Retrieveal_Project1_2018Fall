import torch
from data.dataloader import itemDataset
from torch.utils.data import Dataset,DataLoader

import torch.optim as optim
import torch.nn as nn

from torchvision import transforms, utils
from model.Resnet import resnet

batch_size = 128
check = 10
epoch = 500
num_workers=16

if(torch.cuda.is_available()):
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


def get_data(batch_size):
	train_dataset = itemDataset(file_name='./data/dev_train.csv',transform=transforms.Compose([
															transforms.ToPILImage(),
															transforms.RandomCrop(size=48,padding=3),
															transforms.RandomHorizontalFlip(0.5),
															transforms.ColorJitter(),
															transforms.RandomVerticalFlip(p=0.5),
															transforms.ToTensor(),
															transforms.Normalize((0.5077425080522147,), (0.25500891562522027,)),
														]))
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

	valid_dataset = itemDataset(file_name='./data/dev_valid.csv',transform=transforms.Compose([
															transforms.ToTensor(),
															transforms.Normalize((0.5077425080522147,), (0.25500891562522027,)),
															]))
	valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

	return train_dataloader,valid_dataloader



def main():
	print("loading data")
	train_dataloader,valid_dataloader = get_data(batch_size)


	print("setting model")
	model = resnet()
	model = model.to(device=device)
	print(model)
	optimizer = optim.Adam(model.parameters(),lr=1e-4)
	criterion = nn.CrossEntropyLoss(reduction='sum')
	
	loss_best = 100000000
	print("start training")
	for now in range(epoch):
		print(now)
		loss_sum = 0
		count = 0
		model.train()
		for i,data in enumerate(train_dataloader):
			#first convert the data into cuda
			value = data['value'].to(device = device).view(-1,1,48,48)
			label = data['type'].to(device = device)
			
			out = model(value)
			loss = criterion(out,label) 
			_,pred = torch.topk(out,1)
			pred = pred.view(-1)
			
			#print(label)
			#print(pred)
			count += torch.sum( label==pred )
			
			model.zero_grad()
			loss.backward()
			optimizer.step()
			loss_sum += loss

		if(now%check==0):
			print('*'*10)
			print('training loss:{0} acc:{1}/{2}'.format(loss_sum,count.data,25946))
		
		loss_sum = 0
		count = 0
		model.eval()
		for i,data in enumerate(valid_dataloader):
			with torch.no_grad():
				value = data['value'].to(device = device).view(-1,1,48,48)
				label = data['type'].to(device = device)
			
				out = model(value)
				loss = criterion(out,label) 
				
				_,pred = torch.topk(out,1)
				pred = pred.view(-1)
				#print(label)
				#print(pred)
				count += torch.sum( label==pred )
				loss_sum += loss
		if(now%check==0):
			print('testing loss:{0} acc:{1}/{2}'.format(loss_sum,count,2763))

		if(loss_sum<loss_best):
			torch.save(model.state_dict(), './save_model/last4.pkl')
			loss_best = loss_sum


if(__name__ == '__main__'):
	main()

