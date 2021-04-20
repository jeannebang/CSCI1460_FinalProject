from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
#import unicodedata
import re
from collections import defaultdict
from konlpy.tag import Kkma

class TranslationDataset(Dataset):
    def __init__(self, en_file, ko_file, seq_len, word2id=None):
        """
        Read and parse the translation dataset line by line. 

        :param input_file: the data file pathname
        :param seq_len: sequence length of encoder and decoder
        """
        # read the input file line by line and put the lines in a list.
        # Also unk the words that appear few times.
        en_lines = read_from_corpus(en_file)
        ko_lines = read_from_corpus(ko_file)
        # create the joint vocabulary.
        en_words = list(set([word for line in en_lines for word in line]))
        ko_words = list(set([word for line in ko_lines for word in line]))
        joint_words = list(set(en_words+ko_words))
        pad_token, start_en_token, end_token, mask_token, start_ko_token = "*PAD*", "*START-EN*", "*END*","*MASK*", "*START-KO*"
        aux_tokens = [pad_token, start_en_token, end_token, mask_token, start_ko_token]
            
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
        self.en_enc_input =[] # en input for encoder: "START_EN"+sentence+"END"
        self.en_dec_input =[] # en input for decoder: "START_EN"+sentence
        self.en_dec_output =[] # en input for decoder: sentence+"END"
        self.ko_enc_input =[] # ko input for encoder: "START_KO"+sentence+"END"
        self.ko_dec_input =[] # ko input for decoder: "START_EN"+sentence
        self.ko_dec_output =[] # ko input for decoder: sentence+"END"
        self.len_en_enc =[] # length of encoder input
        self.len_en_dec =[] # length of decoder input = length of encoder input -1
        self.len_ko_enc =[] 
        self.len_ko_dec =[] 
        for l in en_lines:
            ll = [start_en_token] +l +[end_token]
            self.len_en_enc.append(min(len(ll),seq_len))
            self.len_en_dec.append(min(len(ll)-1, seq_len))
            seq=torch.tensor([self.word2id_joint[w] for w in ll])
            self.en_enc_input.append(seq)
            self.en_dec_input.append(seq[:-1])
            self.en_dec_output.append(seq[1:])
        for l in ko_lines:
            ll = [start_ko_token] +l +[end_token]
            self.len_ko_enc.append(min(len(ll),seq_len))
            self.len_ko_dec.append(min(len(ll)-1,seq_len))
            seq=torch.tensor([self.word2id_joint[w] for w in ll])
            self.ko_enc_input.append(seq)
            self.ko_dec_input.append(seq[:-1])
            self.ko_dec_output.append(seq[1:])
        #padding
        self.en_enc_input =pad_input(self.en_enc_input, seq_len)
        self.en_dec_input =pad_input(self.en_dec_input, seq_len)
        self.en_dec_output =pad_input(self.en_dec_output, seq_len)
        self.ko_enc_input =pad_input(self.ko_enc_input, seq_len)
        self.ko_dec_input =pad_input(self.ko_dec_input, seq_len)
        self.ko_dec_output =pad_input(self.ko_dec_output, seq_len)

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # Override method to return length of dataset
        return len(self.len_en_enc)

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
            "en_enc_input": self.en_enc_input[idx],
            "en_dec_input": self.en_dec_input[idx],
            "en_dec_output": self.en_dec_output[idx],
            "ko_enc_input": self.ko_enc_input[idx],
            "ko_dec_input": self.ko_dec_input[idx],
            "ko_dec_output": self.ko_dec_output[idx],
            "len_en_enc": self.len_en_enc[idx],
            "len_en_dec": self.len_en_dec[idx],
            "len_ko_enc": self.len_ko_enc[idx],
            "len_ko_dec": self.len_ko_dec[idx]
        }
        return item


def pad_input(tensor_list, seq_len):
    """
    pad the list of tensors and make it be the given shape.
    
    :param input_tensor: list of tensors, 
    :param seq_len: maximum length of tensor
    
    :return: padded tensor, shape of (len(input_tensor), seq_len)
    """
    pad_tensor = pad_sequence(tensor_list, batch_first=True)
    max_len = pad_tensor.size(1)
    if max_len < seq_len:
        pad_tensor = torch.nn.functional.pad(pad_tensor, pad=(0,seq_len-max_len,0,0),mode='constant',value=0)
    else:
        pad_tensor = pad_tensor[:,:seq_len]
    return pad_tensor


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