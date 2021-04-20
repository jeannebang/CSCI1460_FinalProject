from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
#import unicodedata
import re
from collections import defaultdict
#from konlpy.tag import Kkma

class TranslationDataset(Dataset):
    def __init__(self, en_file, ko_file, seq_len,
                 word2id=None, target_token="*2en*", order=False):
        """
        Read and parse the translation dataset line by line. 

        :param input_file: the data file pathname
        :param seq_len: sequence length of encoder and decoder
        :param target_token: the tag for target language
        :param order: if True, src = eng, tgt = kor
                      if False, src = kor, tgt = eng
        """
        # read the input file line by line and put the lines in a list.
        # Also unk the words that appear few times.
        if order:
            src_lines = read_from_corpus(en_file)
            tgt_lines = read_from_corpus(ko_file)
        else:
            tgt_lines = read_from_corpus(en_file)
            src_lines = read_from_corpus(ko_file)
        # create the joint vocabulary.
        src_words = list(set([word for line in src_lines for word in line]))
        tgt_words = list(set([word for line in tgt_lines for word in line]))
        joint_words = list(set(src_words+tgt_words))
        pad_token, start_token, end_token, mask_token = "*PAD*", "*START*", "*END*","*MASK*"
        aux_tokens = [pad_token, start_token, end_token, mask_token]
        if type(target_token)==str:
            aux_tokens = aux_tokens+[target_token]
            
        if word2id ==None:
            self.word2id_joint = {}       
            curr_id = 0
        else: # append upon word2id
            self.word2id_joint = word2id.copy()
            curr_id = len(word2id)
            
        for w in aux_tokens+joint_words:
            if not w in self.word2id_joint:
                self.word2id_joint[w] = curr_id
                curr_id +=1
        

        # create inputs and labels for both training and validation data
        self.src_input =[] # input for encoder
        self.tgt_input =[] # input for decoder
        self.tgt_output =[] # label, to compare with output of decoder
        self.len_src =[] # length of encoder input
        self.len_tgt =[] # length of decoder input = length of decoder output
        for l in src_lines:
            ll = [target_token] +l
            self.len_src.append(min(len(ll),seq_len))# seq+pad will be forced to have length=seq_len
            seq=torch.tensor([self.word2id_joint[w] for w in ll])
            self.src_input.append(seq)
        for l in tgt_lines:
            ll = [start_token] + l + [end_token] # add start and end tokens
            self.len_tgt.append(min(len(ll)-1,seq_len)) # decoder input = start+l, decoder ouput = l+end
            seq=torch.tensor([self.word2id_joint[w] for w in ll])
            self.tgt_input.append(seq[:-1])
            self.tgt_output.append(seq[1:])
        #padding
        self.src_input = pad_sequence(self.src_input, batch_first=True)
        self.tgt_input = pad_sequence(self.tgt_input, batch_first=True)
        self.tgt_output = pad_sequence(self.tgt_output, batch_first=True)
        max_len_src = max(self.len_src)
        max_len_tgt = max(self.len_tgt)
        
        if max_len_src < seq_len:
            self.src_input = torch.nn.functional.pad(self.src_input, pad=(0,seq_len-max_len_src,0,0),mode='constant',value=0)
        else:
            self.src_input = self.src_input[:,:seq_len]
        if max_len_tgt < seq_len:
            self.tgt_input = torch.nn.functional.pad(self.tgt_input, pad=(0,seq_len-max_len_tgt,0,0),mode='constant',value=0)
            self.tgt_output = torch.nn.functional.pad(self.tgt_output, pad=(0,seq_len-max_len_tgt,0,0),mode='constant',value=0)
        else:
            self.tgt_input = self.tgt_input[:,:seq_len]
            self.tgt_output = self.tgt_output[:,:seq_len]

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # Override method to return length of dataset
        return len(self.len_src)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        item = {
            "src_input": self.src_input[idx],
            "tgt_input": self.tgt_input[idx],
            "tgt_output": self.tgt_output[idx],
            "len_src": self.len_src[idx],
            "len_tgt": self.len_tgt[idx]
        }
        return item


def read_from_corpus(filename):
    lines = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            l = line[:-1] #exclude the last '\n'
            # remove duplicated spaces
            l = re.sub('\s{2,}', ' ', l)
            # split into words
            lines.append(l.split())
    
    return lines
