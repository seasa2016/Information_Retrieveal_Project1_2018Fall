import torch
import torch.nn as nn
import torch.nn.functional as F


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

class bimpm(nn.Module):
	def __init__(self,args):
		super(bimpm,self).__init__()
		self.encoder = Encoder(args)
		self.decoder = Decoder(args)
	
	def forward(self):
		"""
			I will not use this partXDD
		"""
		pass

class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()

		self.args = args
		self.batch_first = args.batch_first

		self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_dim
		self.l = 10

		# ----- Word Representation Layer -----

		self.word_emb = nn.Embedding(args.input_size, args.word_dim, padding_idx=0)


		# ----- Context Representation Layer -----
		self.context_LSTM = nn.LSTM(
			input_size=args.word_dim,
			hidden_size=args.hidden_dim,
			num_layers=args.num_layer,
			batch_first=args.batch_first,
			dropout=args.dropout,
			bidirectional=True
		)

		# ----- Matching Layer -----
		for i in range(1, 9):
			setattr(self, f'mp_w{i}',
					nn.Parameter(torch.rand(self.l, self.args.hidden_dim)))

		# ----- Aggregation Layer -----
		self.aggregation_LSTM = nn.LSTM(
			input_size=self.l * 8,
			hidden_size=args.hidden_dim,
			num_layers=args.num_layer,
			batch_first=args.batch_first,
			dropout=args.dropout,
			bidirectional=True
		)


		self.reset_parameters()
		
	def reset_parameters(self):
		# ----- Word Representation Layer -----
		# zero vectors for padding

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.1, 0.1)

		# ----- Context Representation Layer -----
		nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
		nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
		nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
		nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

		nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
		nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
		nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
		nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

		# ----- Matching Layer -----
		for i in range(1, 9):
			w = getattr(self, f'mp_w{i}')
			nn.init.kaiming_normal_(w)

		# ----- Aggregation Layer -----
		nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
		nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
		nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
		nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

		nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
		nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
		nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
		nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

	def dropout(self, v):
		return F.dropout(v, p=self.args.dropout, training=self.training)

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

			state = [ _[:,desorted_indices] for _ in state ] 

			if(self.batch_first):
				desorted_res = padded_res[desorted_indices]
			else:
				desorted_res = padded_res[:, desorted_indices]

			return desorted_res,state

		# ----- Matching Layer -----
		def mp_matching_func(v1, v2, w):
			"""
			:param v1: (batch, seq_len, hidden_size)
			:param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
			:param w: (l, hidden_size)
			:return: (batch, l)
			"""
			seq_len = v1.shape[1]

			# (1, 1, hidden_size, l)
			w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
			# (batch, seq_len, hidden_size, l)
			print('w',w.shape,'v1',v1.shape)
			v1 = w * torch.stack([v1] * self.l, dim=3)
			if(len(v2.shape) == 3):
				v2 = w * torch.stack([v2] * self.l, dim=3)
			else:
				v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

			# (batch, seq_len, l)
			m = F.cosine_similarity(v1, v2, dim=2)

			return m

		def mp_matching_func_pairwise(v1, v2, w,mask):
			"""
			this is for the attention

			:param v1: (batch, seq_len1, hidden_size)
			:param v2: (batch, seq_len2, hidden_size)
			:param w: (l, hidden_size)
			:return: (batch, l, seq_len1, seq_len2)
			"""
			# (1, l, 1, hidden_size)
			w = w.unsqueeze(0).unsqueeze(2)
			# (batch, l, seq_len, hidden_size)
			v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
			# (batch, l, seq_len, hidden_size->1)
			v1_norm = v1.norm(p=2, dim=3, keepdim=True)
			v2_norm = v2.norm(p=2, dim=3, keepdim=True)

			# (batch, l, seq_len1, seq_len2)
			n = torch.matmul(v1, v2.transpose(2, 3))
			d = v1_norm * v2_norm.transpose(2, 3)

			# (batch, seq_len1, seq_len2, l)
			m = div_with_small_value(n, d).permute(0, 2, 3, 1)
			
			final_mask = torch.matmul(mask['query'].float().transpose(0,1),mask['answer'].float())
			final_mask = final_mask>0.5

			m[:,final_mask,:] = 0

			return m

		def attention(v1, v2,mask):
			"""
			:param v1: (batch, seq_len1, hidden_size)
			:param v2: (batch, seq_len2, hidden_size)
			:return: (batch, seq_len1, seq_len2)
			"""

			# (batch, seq_len1, 1)
			v1_norm = v1.norm(p=2, dim=2, keepdim=True)
			# (batch, 1, seq_len2)
			v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

			# (batch, seq_len1, seq_len2)
			a = torch.bmm(v1, v2.permute(0, 2, 1))
			d = v1_norm * v2_norm

			m = div_with_small_value(a, d)
			
			final_mask = torch.matmul(mask['query'].transpose(0,1).float(),mask['answer'].float())
			final_mask = final_mask>0.5

			m[:,final_mask] = 0

			return m

		def div_with_small_value(n, d, eps=1e-8):
			# too small values are replaced by 1e-8 to prevent it from exploding.
			d = d * (d > eps).float() + eps * (d <= eps).float()
			return n / d

		# ----- Word Representation Layer -----
		# (batch, seq_len) -> (batch, seq_len, word_dim)

		mask = {}
		mask['query'] =  query.eq(0)
		mask['answer'] =  answer.eq(0)

		query = self.word_emb(query)
		answer = self.word_emb(answer)

		if self.args.use_char_emb:
			# (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
			seq_len_p = kwargs['char_p'].size(1)
			seq_len_h = kwargs['char_h'].size(1)

			char_p = kwargs['char_p'].view(-1, self.args.max_word_len)
			char_h = kwargs['char_h'].view(-1, self.args.max_word_len)

			# (batch * seq_len, max_word_len, char_dim)-> (1, batch * seq_len, char_hidden_size)
			_, (char_p, _) = self.char_LSTM(self.char_emb(char_p))
			_, (char_h, _) = self.char_LSTM(self.char_emb(char_h))

			# (batch, seq_len, char_hidden_size)
			char_p = char_p.view(-1, seq_len_p, self.args.char_hidden_size)
			char_h = char_h.view(-1, seq_len_h, self.args.char_hidden_size)

			# (batch, seq_len, word_dim + char_hidden_size)
			p = torch.cat([p, char_p], dim=-1)
			h = torch.cat([h, char_h], dim=-1)

		query = self.dropout(query)
		answer = self.dropout(answer)

		# ----- Context Representation Layer -----
		# (batch, seq_len, hidden_size * 2)
		packed_inputs,desorted_indices = pack(query,query_len)
		res, state = self.context_LSTM(packed_inputs)
		query_res,_ = unpack(res, state,desorted_indices)

		packed_inputs,desorted_indices = pack(answer,answer_len)
		res, state = self.context_LSTM(packed_inputs)
		answer_res,_ = unpack(res, state,desorted_indices)

		query_res = self.dropout(query_res)
		answer_res = self.dropout(answer_res)

		# (batch, seq_len, hidden_size)
		query_fw, query_bw = torch.split(query_res, self.args.hidden_dim, dim=-1)
		answer_fw, answer_bw = torch.split(answer_res, self.args.hidden_dim, dim=-1)

		# 1. Full-Matching

		# (batch, seq_len, hidden_size), (batch, hidden_size)
		# -> (batch, seq_len, l)
		query_full_fw = mp_matching_func(query_fw, answer_fw[ torch.tensor(list( range( answer_len.shape[0] ) )).view(-1,1) , answer_len.view(-1,1)-1 , :], self.mp_w1)
		query_full_bw = mp_matching_func(query_bw, answer_bw[:,  0, :], self.mp_w2)
		answer_full_fw = mp_matching_func(answer_fw, query_fw[ torch.tensor(list( range( query_len.shape[0] ) )).view(-1,1) , query_len.view(-1,1)-1 , :], self.mp_w1)
		answer_full_bw = mp_matching_func(answer_bw, query_bw[:,  0, :], self.mp_w2)

		# 2. Maxpooling-Matching
		# (batch, seq_len1, seq_len2, l)
		mv_max_fw = mp_matching_func_pairwise(query_fw, answer_fw, self.mp_w3,mask)
		mv_max_bw = mp_matching_func_pairwise(query_bw, answer_bw, self.mp_w4,mask)

		# (batch, seq_len, l)
		query_max_fw, _ = mv_max_fw.max(dim=2)
		query_max_bw, _ = mv_max_bw.max(dim=2)
		answer_max_fw, _ = mv_max_fw.max(dim=1)
		answer_max_bw, _ = mv_max_bw.max(dim=1)

		# 3. Attentive-Matching

		# (batch, seq_len1, seq_len2)
		att_fw = attention(query_fw, answer_fw,mask)
		att_bw = attention(query_bw, answer_bw,mask)

		# (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
		# (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
		# -> (batch, seq_len1, seq_len2, hidden_size)
		answer_fw = answer_fw.unsqueeze(1) * att_fw.unsqueeze(3)
		answer_bw = answer_bw.unsqueeze(1) * att_bw.unsqueeze(3)
		# (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
		# (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
		# -> (batch, seq_len1, seq_len2, hidden_size)
		query_fw = query_fw.unsqueeze(2) * att_fw.unsqueeze(3)
		query_bw = query_bw.unsqueeze(2) * att_bw.unsqueeze(3)

		# (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
		answer_mean_fw = div_with_small_value(answer_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
		answer_mean_bw = div_with_small_value(answer_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

		# (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
		query_mean_fw = div_with_small_value(query_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
		query_mean_bw = div_with_small_value(query_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

		# (batch, seq_len, l)
		query_att_mean_fw = mp_matching_func(query_fw, answer_mean_fw, self.mp_w5)
		query_att_mean_bw = mp_matching_func(query_bw, answer_mean_bw, self.mp_w6)
		answer_att_mean_fw = mp_matching_func(answer_fw, query_mean_fw, self.mp_w5)
		answer_att_mean_bw = mp_matching_func(answer_bw, query_mean_bw, self.mp_w6)

		# 4. Max-Attentive-Matching

		# (batch, seq_len1, hidden_size)
		answer_fw[mask['answer'],:] = 0
		answer_bw[mask['answer'],:] = 0
		query_fw[mask['query'],:] = 0
		query_bw[mask['query'],:] = 0

		answer_max_fw, _ = answer_fw.max(dim=2)
		answer_max_bw, _ = answer_bw.max(dim=2)
		# (batch, seq_len2, hidden_size)
		query_max_fw, _ = query_fw.max(dim=1)
		query_max_bw, _ = query_bw.max(dim=1)

		# (batch, seq_len, l)
		query_att_max_fw = mp_matching_func(query_fw, answer_max_fw, self.mp_w7)
		query_att_max_bw = mp_matching_func(query_bw, answer_max_bw, self.mp_w8)
		answer_att_max_fw = mp_matching_func(answer_fw, query_max_fw, self.mp_w7)
		answer_att_max_bw = mp_matching_func(answer_bw, query_max_bw, self.mp_w8)

		# (batch, seq_len, l * 8)
		query = torch.cat(
			[query_full_fw, query_max_fw, query_att_mean_fw, query_att_max_fw,
				query_full_bw, query_max_bw, query_att_mean_bw, query_att_max_bw], dim=2)
		answer = torch.cat(
			[answer_full_fw, answer_max_fw, answer_att_mean_fw, answer_att_max_fw,
				answer_full_bw, answer_max_bw, answer_att_mean_bw, answer_att_max_bw], dim=2)

		query = self.dropout(query)
		answer = self.dropout(answer)

		# ----- Aggregation Layer -----
		# (batch, seq_len, l * 8) -> (2, batch, hidden_size)


		packed_inputs,desorted_indices = pack(query,query_len)
		res, state = self.aggregation_LSTM(packed_inputs)
		query_res,(agg_query_last, _) = unpack(res, state,desorted_indices)

		packed_inputs,desorted_indices = pack(answer,answer_len)
		res, state = self.aggregation_LSTM(packed_inputs)
		answer_res,(agg_answer_last, _) = unpack(res, state,desorted_indices)


		# 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
		x = torch.cat(
			[agg_query_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2),
				agg_answer_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)], dim=1)
		x = self.dropout(x)


		return x
