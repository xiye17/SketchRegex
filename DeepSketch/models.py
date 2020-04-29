import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var

import numpy as np
import math

class BatchDataLoader(object):

    def __init__(self, data, all_in, all_out, batch_size=1, shuffle=False, drop_last=False, return_id=False):
        self.data = data
        self.all_in = torch.from_numpy(all_in).long()
        self.all_in_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in data]))
        self.all_out = torch.from_numpy(all_out).long()
        self.all_out_lens = torch.from_numpy(np.asarray([len(ex.y_indexed) for ex in data]))
        self.all_id = torch.from_numpy(np.asarray([ex.id for ex in data]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_cnt = 0
        self.max_batch = 0
        self.sampler = None
        self.return_id = return_id

    def __len__(self):
        return int(math.ceil(len(self.data) * 1.0 / self.batch_size))

    def __next__(self):
        if self.batch_cnt == self.max_batch:
            # self.__iter__()
            raise StopIteration

        b_start = self.batch_cnt * self.batch_size
        b_end = min(b_start + self.batch_size, len(self.data))
        b_idx = self.sampler[b_start:b_end]
        b_idx = np.sort(b_idx)

        b_in = self.all_in[b_idx]
        b_in_lens = self.all_in_lens[b_idx]
        # print(b_in)
        b_out = self.all_out[b_idx]
        b_out_lens = self.all_out_lens[b_idx]
        self.batch_cnt += 1
        if self.return_id:
            b_ids = self.all_id[b_idx]
            return b_in, b_in_lens, b_out, b_out_lens, b_ids
        else:
            return b_in, b_in_lens, b_out, b_out_lens

    def __iter__(self):
        self.batch_cnt = 0
        self.max_batch = self.__len__()
        if self.shuffle:
            self.sampler = np.random.choice(len(self.data), len(self.data), replace=False)
            # print(self.sampler)
        else:
            self.sampler = np.arange(len(self.data))
        return self

def sent_lens_to_mask(lens, max_length):
    mask = torch.BoolTensor(np.asarray([[1 if j < lens.data[i].item() else 0
                                        for j in range(0, max_length)] for i in range(0, lens.shape[0])]))
    # match device of input
    return mask.to(lens.device)

def get_inf_mask(mask):
    inf_mask = torch.zeros_like(mask, dtype=torch.float32)
    inf_mask.masked_fill_(~mask, float("-inf"))
    return inf_mask

# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = sent_lens_to_mask(input_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        # print(max_length, output.size(), h_t[0].size(), h_t[1].size())

        return (output, context_mask, h_t)


# decoder
# in: word_vec (input), hidden_layer
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, voc_size, dropout):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size

        self.reduce_h_v = nn.Linear(hidden_size, voc_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)

    def forward(self, embedded_words, hidden_states):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        outputs, hn = self.rnn(embedded_words, hidden_states)
        voc_scores = self.reduce_h_v(outputs)
        voc_scores = voc_scores.reshape((-1, self.voc_size))
        voc_scores = F.softmax(voc_scores, 1)
        return voc_scores, hn

class LuongAttention(nn.Module):

    def __init__(self, hidden_size, context_size=None):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = hidden_size if context_size is None else context_size
        self.attn = torch.nn.Linear(self.context_size, self.hidden_size)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.attn.weight, gain=1)
        nn.init.constant_(self.attn.bias, 0)

    # input query: batch * q * hidden, contexts: batch * c * hidden
    # output: batch * len * q * c
    def forward(self, query, context, inf_mask=None, requires_weight=False):
        # Calculate the attention weights (energies) based on the given method
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)

        e = self.attn(context)
        # e: B * Q * C
        e = torch.matmul(query, e.transpose(1, 2))
        if inf_mask is not None:
            e = e + inf_mask.unsqueeze(1)

        # dim w: B * Q * C, context: B * C * H, wanted B * Q * H
        w = F.softmax(e, dim=2)
        c = torch.matmul(w, context)
        # # Return the softmax normalized probability scores (with added dimension
        if requires_weight:
            return c.transpose(0, 1), w
        return c.transpose(0, 1)

# decoder
# in: word_vec (input), hidden_layer
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class AttnRNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, context_hidden_size, voc_size, dropout):
        super(AttnRNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.context_hidden_size = context_hidden_size
        self.attn = LuongAttention(hidden_size, context_hidden_size)

        self.reduce_h_v = nn.Linear(hidden_size + context_hidden_size, voc_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)

    def forward(self, embedded_words, hidden_states, context_states, context_inf_mask):

        outputs, hn = self.rnn(embedded_words, hidden_states)

        # attn_weights: batch * len
        # context_states batch * len * contedxt_hidden_size
        # contexts = torch.bmm(attn_weights.unsqueeze(1), context_states.transpose(0, 1))
        # output_contexts = contexts.view((1, -1, self.context_hidden_size))
        output_contexts = self.attn(hn[0], context_states, inf_mask=context_inf_mask)
        concated_outpts = torch.cat((outputs, output_contexts), 2)
        # concated_outpts = outputs
        # concated_outpts = F.relu(concated_outpts)
        voc_scores = self.reduce_h_v(concated_outpts)

        voc_scores = voc_scores.reshape((-1, self.voc_size))
        voc_scores = F.softmax(voc_scores, 1)
        return voc_scores, hn
