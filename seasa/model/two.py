import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class Encoder(nn.Module):
	def __init__(self,args):
		super(Encoder,self).__init__()

		self.input_size=args.input_size
		
		self.word_dim = args.word_dim
		self.hidden_dim =args.hidden_dim
		
		self.num_layers=args.num_layer
		self.batch_first=args.batch_first
		self.dropout=args.dropout


		self.word_embedding = nn.Embedding(args.input_size,args.word_dim,padding_idx=0)
		self.word_rnn = nn.LSTM(
			input_size=args.word_dim,
			hidden_size=args.hidden_dim,
			num_layers=args.num_layer,
			batch_first=args.batch_first,
			dropout=args.dropout,
			bidirectional=True
		)
		self.aggre_rnn = nn.LSTM(
			input_size=args.word_dim*2,
			hidden_size=args.hidden_dim,
			num_layers=args.num_layer,
			batch_first=args.batch_first,
			dropout=args.dropout,
			bidirectional=True
		)

		self.dropout = nn.Dropout(p=0.5)
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
			state = [ _ for _ in state]

			for i in range(len(state)):
				state[i] = state[i][:,desorted_indices]
			
			if(self.batch_first):
				desorted_res = padded_res[desorted_indices]
			else:
				desorted_res = padded_res[:, desorted_indices]

			return desorted_res,state
			
		def run_rnn(rnn,seq_emb,seq_length):
			#query part
			packed_inputs,desorted_indices = pack(seq_emb,seq_length)
			res, state = rnn(packed_inputs)
			res,state = unpack(res, state,desorted_indices)

			return res,state

		def feat_extract(query_output,query_length,answer_output,answer_length,mask):
			"""
			answer_output: batch*sentence*feat_len
			query_output:  batch*sentence*feat_len
			"""
			query_length = query_length.float()
			answer_length = answer_length.float()

			query_length = query_length.view(-1,1)
			answer_length = answer_length.view(-1,1)
			if(self.batch_first):
				query_output = query_output.sum(dim=1)
				answer_output = answer_output.sum(dim=1)
			else:
				query_output = query_output.sum(dim=0)
				answer_output = answer_output.sum(dim=0)

			query_normal = query_output.div(query_length).div(math.sqrt(float(self.hidden_dim)))
			answer_output = query_output.div(answer_length).div(math.sqrt(float(self.hidden_dim)))


			return query_output,query_normal

		def attention(query_output,query_length,answer_output,answer_length,mask):
			"""
			answer_output: batch*sent_ans*feat_len
			query_output:  batch*sent_q*feat_len
			"""
			if(not self.batch_first):
				answer_output = answer_output.transpose(0,1) 
			
			# batch*feat_len*1
			query_normal = query_output.div(math.sqrt(float(self.hidden_dim))).transpose(1,2)

			#get the attention
			#batch*sent_ans*sent_qu
			attn_weight = answer_output.bmm(query_normal)
		   
		   	
			final_mask = mask['answer'].unsqueeze(2).float().bmm(mask['query'].unsqueeze(1).float())>0.5
			
			#perform mask for the padding data
			attn_weight[final_mask] = -1e8
			
			#batch*sent_ans*sent_qu
			attn_weight_qu = attn_weight.softmax(dim=-1)
			attn_weight_ans = attn_weight.softmax(dim=-2)

			answer_extract = attn_weight_qu.bmm(query_output)
			query_extract = (answer_output.transpose(1,2)).bmm(attn_weight_ans).transpose(1,2)

			return query_extract,answer_extract
		
		#first check for the mask ans the embedding
		mask = {}
		mask['answer'] =  answer.eq(0)
		mask['query'] =  query.eq(0)

		query_emb = self.word_embedding(query)
		answer_emb = self.word_embedding(answer)


		query_res,_ = run_rnn(self.word_rnn,query_emb,query_len)
		answer_res,_ = run_rnn(self.word_rnn,answer_emb,answer_len)

		#extract the representation of the sentence
		query_temp,answer_temp = attention(query_res,query_len,answer_res,answer_len,mask)

		
		query_res,_ = run_rnn(self.aggre_rnn,query_temp,query_len)
		answer_res,_ = run_rnn(self.aggre_rnn,answer_temp,answer_len)

		query_result,answer_result = feat_extract(query_res,query_len,answer_res,answer_len,mask)

		total_output = torch.cat([query_result,answer_result],dim=-1)

		return total_output


class Decoder(nn.Module):
	def __init__(self,args):
		super(Decoder,self).__init__()
		self.linear1 = nn.Linear(4*args.hidden_dim,args.hidden_dim)
		self.linear2 = nn.Linear(args.hidden_dim,1)

		self.act = nn.Sigmoid()
	def forward(self,x):
		"""
			here I will simply use two layer feedforward
		"""
		out = self.linear1(x)
		out = self.act(out)

		out = self.linear2(out)
		
		return out

class two(nn.Module):
	def __init__(self,args):
		super(two,self).__init__()

		self.encoder = Encoder(args)
		self.decoder = Decoder(args)
	
	def forward(self):
		"""
			I will not use this partXDD
		"""
		pass
