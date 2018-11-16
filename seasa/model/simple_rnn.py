import torch
import torch.nn as nn
import torch.nn.functional as f

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder,self).__init__()

        self.lstm =  = nn.LSTM(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
    def forward(self,query,query_len,answer,answer_len):
        pass


class Dncoder(nn.Module):
    def __init__(self,args):
        super(Dncoder,self).__init__()
        self.linear1 = nn.Linear(4*args.hidden,args.hidden)
        self.linear2 = nn.Linear(args.hidden,1)

        self.act = nn.Tanh()
        self.output_act = nn.Sigmoid()
    def forward(self,x):
        """
            here I will simply use two layer feedforward
        """
        out = self.linear1(x)
        out = self.act(out)

        out = self.linear2(out)
        out = self.output_act(out)
        
        return out

class Rank(nn.Module):
    def __init__(self,args):
        super(Rank,self).__init__()

        self.linear1 = nn.Linear(8*args.hidden,args.hidden)
        self.linear2 = nn.Linear(args.hidden,1)

        self.act = nn.Tanh()
        self.output_act = nn.Sigmoid()
    def forward(self,left_x,right_x):
        """
            here I will simply use two layer feedforward
        """
        x = torch.cat(left_x,right_x,dim=-1)
        out = self.linear1(x)
        out = self.act(out)

        out = self.linear2(out)
        out = self.output_act(out)
        
        return out

class simple_rnn(nn.Module):
    def __init__(self,args):
        super(simple_rnn,self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Dncoder(args)
        self.rank = Rank(args)
    
    def forward(self):
        """
            I will not use this partXDD
        """
        pass