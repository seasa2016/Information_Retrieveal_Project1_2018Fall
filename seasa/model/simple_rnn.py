import torch
import torch.nn as nn
import torch.nn.functional as f

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder,self).__init__()

        self.input_size=args.input_size,
        self.hidden_size=args.hidden_size,
        self.num_layers=args.num_layers,
        self.batch_first=args.batch_first,
        self.dropout=args.dropout,


        self.word_embedding = nn.Embedding(args.input_size,args.word_dim)
        self.rnn = nn.LSTM(
            input_size=args.word_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=args.batch_first,
            dropout=args.dropout,
            bidirectional=True
        )
    def forward(self,query,query_len,answer,answer_len):
        def pack(seq,seq_length):
            sorted_seq_lengths, indices = torch.sort(seq_length, descending=True)
            _, desorted_indices = torch.sort(indices, descending=False)

            if self.batch_first:
                seq = seq[indices]
            else:
                seq = seq[:, indices]
            packed_inputs = nn.utils.rnn.pack_padded_sequence(seq,
                                                            sorted_seq_lengths.cpu().numpy(),
                                                            batch_first=self.batch_first)

            return packed_inputs,desorted_indices

        def unpack(res, state,desorted_indices):
            padded_res,_ = nn.utils.rnn.pad_packed_sequence(res, batch_first=self.batch_first)
            
            state = state[desorted_indices]
            if(self.batch_first):
                desorted_res = padded_res[desorted_indices]
            else:
                desorted_res = padded_res[:, desorted_indices]

            return desorted_res,state
            
        def feat_extract(query_output,query_length,answer_output,answer_length,mask):
            """
            answer_output: batch*sentence*feat_len
            query_output:  batch*sentence*feat_len
            """
            query_length = query_length.view(-1,1)
            if(self.batch_first):
                query_output = query_output.sum(dim=1)
            else:
                query_output = query_output.sum(dim=0)
                answer_output = answer_output.transpose(0,1) 
            query_normal = query_output.div(query_length).div(query_length.sqrt()).unsqueeze(2)        # batch*feat_len*1

            #get the attention
            attn_weight = answer_output.bmm(query_normal).squeeze(2)         #batch*sentence*1
            #perform mask for the padding data
            attn_weight[mask] = 1e-8
            attn_weight = attn_weight.softmax(dim=-1)       #batch*sentence*1

            answer_extract = (query_output.transpose(1,2)).bmm(attn_weight).squeeze(2)

            return query_output,answer_extract
        
        #first check for the mask ans the embedding
        mask =  query.eq(0)

        query_emb = self.word_embedding(query)
        answer_emb = self.word_embedding(answer)

        #query part
        packed_inputs,desorted_indices = pack(query_emb,query_len)
        res, state = self.rnn(packed_inputs)
        query_res,_ = unpack(res, state,desorted_indices)
        
        #answer part
        packed_inputs,desorted_indices = pack(answer_emb,answer_len)
        res, state = self.rnn(packed_inputs)
        answer_res,_ = unpack(res, state,desorted_indices)

        #extract the representation of the sentence
        query_result,answer_result = feat_extract(query_res,query_len,answer_res,answer_len,mask)

        total_output = torch.cat([query_result,answer_result],dim=-1)

        return total_output


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