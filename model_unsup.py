from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, seq_len,
                 mask_token_id):
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
        self.mask_token_id = mask_token_id
            
        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding_joint = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn_enc = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True, bidirectional=True)
        self.rnn_dec = nn.GRU(self.embedding_size+self.rnn_size*2, self.rnn_size, batch_first=True)
        self.att = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.linear = nn.Linear(self.rnn_size, self.vocab_size) # linear layer convert hidden_dec into output_vocab
        
    def add_noise(self,x,p=0.1,k=3, mask_token_id=3):
        """
        Add noise on the sequence.
        
        :param x: The sequence, (batch_size, seq_len)
        :param p: probability of dropping the words
        :param k: parameter for slightly shuffle the words.
        """
        mask = (torch.rand(x.shape)<p) #bool tensor, 1 for dropped words, 0 for remain words
        x[mask] = mask_token_id * torch.ones(x.shape).type(x.type())[mask] #dropped words id => mask_id
        # slightly permute as in paper.
        q = torch.arange(x.size(1)).expand(x.shape) + torch.rand(x.shape) * (k+1)
        sorted_dim1 = torch.argsort(q)
        # sorted_dim0 = [[0,...,0],[1,...,1],...,[n,...,n]], n=batch_size
        sorted_dim0 = torch.arange(x.size(0)).reshape((-1,1)).repeat(1,x.size(1))
        output = x[sorted_dim0, sorted_dim1]
        return output
       
    def encoder(self, encoder_inputs, encoder_lengths):
        """
        encoder layer.

        :param encoder_inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs

        :return: hidden state of encoder, (batch_size, seq_len, rnn_size*2)
        """
        embed_enc = self.embedding_joint(encoder_inputs) # (batch_size, seq_len, embedding_size)

        packed_inputs_enc = pack_padded_sequence(embed_enc, encoder_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs_enc, hidden_enc = self.rnn_enc(packed_inputs_enc) 
        enc_outputs, enc_output_lengths = pad_packed_sequence(packed_outputs_enc, batch_first=True)# (batch_size, max(encoder_lengths), rnn_size*2)
        pad_enc_outputs = F.pad(enc_outputs, pad=(0,0,0,max(0,self.seq_len-enc_outputs.size()[1]),0,0),mode='constant',value=0) # (batch_size, seq_len, rnn_size*2)
        return pad_enc_outputs
    
    def decoder(self, decoder_inputs, decoder_lengths, hidden_states):
        """
        decoder layer.

        :param decoder_inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs
        :param hidden_states: to be concatenated with embeddings and put into decoder.
                              can be interperetated as attention, or hidden states from encoder.
                              should be size of (batch_size, seq_len, rnn_size*2)

        :return: logits for next word, (batch_size, seq_len, vocab_size)
        """
        embed_dec = self.embedding_joint(decoder_inputs) # (batch_size, seq_len, embedding_size)
        emb_n_att = torch.cat((hidden_states, embed_dec),2) # (batch_size, seq_len, rnn*2+embed)
        packed_inputs_dec = pack_padded_sequence(emb_n_att, decoder_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs_dec, hidden_dec = self.rnn_dec(packed_inputs_dec)
        dec_outputs, dec_output_lengths = pad_packed_sequence(packed_outputs_dec, batch_first=True) # (batch_size, max(decoder_lengths), rnn_size)
        pad_dec_outputs = F.pad(dec_outputs, pad=(0,0,0,max(0,self.seq_len-dec_outputs.size()[1]),0,0),mode='constant',value=0) # (batch_size, seq_len, rnn_size)
        outputs = self.linear(pad_dec_outputs)# (batch_size, seq_len, vocab_size)
        return outputs 
      
    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths,
                decoder_lengths,mask_token_id=3):

        """
        Runs the forward pass of the model. autoencoder.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs
        :param mask_token_id: mask token id used for adding noise

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """
        with torch.no_grad(): # is it necessary???
            encoder_inputs_noise = self.add_noise(encoder_inputs,mask_token_id = mask_token_id)
        pad_enc_outputs = self.encoder(encoder_inputs_noise, encoder_lengths)
        attention = self.att(pad_enc_outputs.transpose(1,2)).transpose(1,2) # (batch_size, seq_len, rnn_size*2)
        outputs = self.decoder(decoder_inputs, decoder_lengths, attention)
        return outputs

    def translate(self, encoder_inputs, encoder_lengths, start_token_id=1, end_token_id=2):
        """
        translate the sentence. the output starts with start_token_id.

        :param encoder_inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param start_token_id: the output starts with start_token_id.
        :param end_token_id: end token id.
        
        :return: the translated token ids (batch_size, seq_len) and lengths.
        """
        batch_size, seq_len = encoder_inputs.size()
        enc_outputs = self.encoder(encoder_inputs, encoder_lengths)
        attention = self.att(enc_outputs.transpose(1,2)).transpose(1,2) # (batch_size, seq_len, rnn_size*2)
   
        dec_outputs = torch.zeros_like(encoder_inputs) #(batch_size, seq_len)
        # the first token of output always starts with start_token_id
        dec_outputs[:,0] = torch.ones(batch_size,dtype=torch.int) * start_token_id
        dec_lengths = torch.ones(batch_size, dtype=torch.int)
        end_flag = torch.ones(batch_size,dtype=torch.int)
        
        for i in range(seq_len-1):
            output = self.decoder(dec_outputs, dec_lengths, attention) # (batch_size, seq_len, vocab_size)
            next_id = torch.argmax(output[:,i,:],dim=1) # (batch_size)
            #update
            dec_outputs[:,i+1] = next_id * end_flag.to(device)
            dec_lengths = dec_lengths + end_flag
            end_flag[next_id==end_token_id] = torch.zeros(batch_size,dtype=torch.int)[next_id==end_token_id]
            #if last token is "end" for all sentences in a batch, stop iteration
            if torch.sum(end_flag) ==0:
                break
        return dec_outputs, dec_lengths
    
    def translate_teacher_forcing(self, encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths, start_token_id=1, end_token_id=2):
        """
        translate the sentence. the output starts with start_token_id.

        :param encoder_inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param start_token_id: the output starts with start_token_id.
        :param end_token_id: end token id.
        
        :return: the translated token ids (batch_size, seq_len) and lengths.
        """
        batch_size, seq_len = encoder_inputs.size()
        enc_outputs = self.encoder(encoder_inputs, encoder_lengths)
        attention = self.att(enc_outputs.transpose(1,2)).transpose(1,2) # (batch_size, seq_len, rnn_size*2)
        dec_outputs = self.decoder(decoder_inputs, decoder_lengths, attention)
        return dec_outputs, decoder_lengths
