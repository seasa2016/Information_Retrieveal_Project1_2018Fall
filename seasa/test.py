import torch
from data.dataloader import itemDataset
from torch.utils.data import Dataset,DataLoader
import sys
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
    test_dataset = itemDataset(file_name='./data/test.csv',transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5077425080522147,), (0.25500891562522027,)),
								                        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return test_dataloader



def main():
    print("loading data")
    test_dataloader = get_data(batch_size)


    print("setting model")
    model = resnet()
    model.load_state_dict(torch.load(sys.argv[1]))

    model = model.to(device=device)
    

    print("start training")
    
    model.eval()
    with open(sys.argv[2],'w') as f:
        f.write('id,label\n')
        for i,data in enumerate(test_dataloader):
            with torch.no_grad():
                value = data['value'].to(device = device).view(-1,1,48,48)
                label = data['type'].cpu().view(-1)
            
                out = model(value)
                
                _,pred = torch.topk(out,1)
                pred = pred.cpu().view(-1)

                for i in range(label.shape[0]):
                    f.write('{0},{1}\n'.format(label[i].item(),pred[i].item()))
    


if(__name__ == '__main__'):
    main()

