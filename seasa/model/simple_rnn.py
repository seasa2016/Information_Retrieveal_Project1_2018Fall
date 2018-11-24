import torch
import torch.nn as nn
import torch.nn.functional as f

class Encoder(nn.Module):
	def __init__(self,args):
		super(Encoder,self).__init__()

		self.input_size = args.input_size
		self.hidden_size = args.hidden_dim
		self.num_layer = args.num_layer
		self.batch_first = args.batch_first
		self.dropout = args.dropout


		self.word_embedding = nn.Embedding(args.input_size,args.word_dim)
		self.rnn = nn.LSTM(
			input_size=args.word_dim,
			hidden_size=args.hidden_dim,
			num_layers=args.num_layer,
			batch_first=args.batch_first,
			dropout=args.dropout,
			bidirectional=True
		)

		self.dropout = nn.Dropout(0.5)
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

			state = [state[i][:,desorted_indices] for i in range(len(state)) ] 
			
			if(self.batch_first):
				desorted_res = padded_res[desorted_indices]
			else:
				desorted_res = padded_res[:, desorted_indices]

			return desorted_res,state
			
		def feat_extract(query_output,query_length,answer_output,answer_length,mask):
			"""
			answer_output: batch*sentence*feat_len
			query_output:  batch*sentence*feat_len


			for simple rnn, we just take the output from 
			"""
			if( self.batch_first == False ):
				answer_output = answer_output.transpose(0,1) 
				query_output = query_output.transpose(0,1) 

			query_output = [torch.cat([ query_output[i][ query_length[i]-1 ][:self.hidden_size] , 
										query_output[i][0][self.hidden_size:]] , dim=-1 ) for i in range(query_length.shape[0])]
			query_output = torch.stack(query_output,dim=0)

			answer_output = [torch.cat([ answer_output[i][ answer_length[i]-1 ][:self.hidden_size] , 
										answer_output[i][0][self.hidden_size:]] , dim=-1 ) for i in range(answer_length.shape[0])]
			answer_output = torch.stack(answer_output,dim=0)



			return query_output,answer_output
		
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
		query_result,answer_result = feat_extract(query_res,query_len.int(),answer_res,answer_len.int(),mask)

		total_output = torch.cat([query_result,answer_result],dim=-1)

		return total_output


class Decoder(nn.Module):
	def __init__(self,args):
		super(Decoder,self).__init__()
		self.linear1 = nn.Linear(4*args.hidden_dim,args.hidden_dim)
		self.linear2 = nn.Linear(args.hidden_dim,1)

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

class simple_rnn(nn.Module):
	def __init__(self,args):
		super(simple_rnn,self).__init__()
		self.encoder = Encoder(args)
		self.decoder = Decoder(args)
	
	def forward(self):
		"""
			I will not use this partXDD
		"""
		pass
