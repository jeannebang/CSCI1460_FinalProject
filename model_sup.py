from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, seq_len):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the input
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        :param seq_len: The sequence length of encoder and decoder
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size=vocab_size
        self.rnn_size=rnn_size
        self.embedding_size=embedding_size
        self.seq_len = seq_len
            
        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding_joint = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn_enc = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True, bidirectional=True)
        self.rnn_dec = nn.GRU(self.embedding_size+self.rnn_size*2, self.rnn_size, batch_first=True)
        self.att = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.linear = nn.Linear(self.rnn_size, self.vocab_size) # linear layer convert hidden_dec into output_vocab

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths,
                decoder_lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """
        # TODO: write forward propagation
        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
        embed_enc = self.embedding_joint(encoder_inputs) # (batch_size, seq_len, embedding_size)
        embed_dec = self.embedding_joint(decoder_inputs) # (batch_size, seq_len, embedding_size)
        
        packed_inputs_enc = pack_padded_sequence(embed_enc, encoder_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs_enc, hidden_enc = self.rnn_enc(packed_inputs_enc) 
        enc_outputs, enc_output_lengths = pad_packed_sequence(packed_outputs_enc, batch_first=True)# (batch_size, max(encoder_lengths), rnn_size*2)
        pad_enc_outputs = F.pad(enc_outputs, pad=(0,0,0,max(0,self.seq_len-enc_outputs.size()[1]),0,0),mode='constant',value=0) # (batch_size, seq_len, rnn_size*2)
        attention = self.att(pad_enc_outputs.transpose(1,2)).transpose(1,2) # (batch_size, seq_len, rnn_size*2)
        
        emb_n_att = torch.cat((attention, embed_dec),2) # (batch_size, seq_len, rnn*2+embed)
        packed_inputs_dec = pack_padded_sequence(emb_n_att, decoder_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs_dec, hidden_dec = self.rnn_dec(packed_inputs_dec)
        dec_outputs, dec_output_lengths = pad_packed_sequence(packed_outputs_dec, batch_first=True) # (batch_size, max(decoder_lengths), rnn_size)
        pad_dec_outputs = F.pad(dec_outputs, pad=(0,0,0,max(0,self.seq_len-dec_outputs.size()[1]),0,0),mode='constant',value=0) # (batch_size, seq_len, rnn_size)
        outputs = self.linear(pad_dec_outputs)# (batch_size, seq_len, output_size)
        return outputs

