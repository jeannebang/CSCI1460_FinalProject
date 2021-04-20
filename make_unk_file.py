###NOTE: Make sure there is no additional empty line at the end of the output file.

import argparse
#import unicodedata
import re
from collections import defaultdict
from konlpy.tag import Kkma
from tqdm import tqdm

def count_vocabs(eng_lines, ko_lines):
    eng_vocab = defaultdict(lambda: 0)
    ko_vocab = defaultdict(lambda: 0)

    for eng_line in eng_lines:
        for eng_word in eng_line:
            eng_vocab[eng_word] += 1
    for ko_line in ko_lines:
        for ko_word in ko_line:
            ko_vocab[ko_word] += 1

    return eng_vocab, ko_vocab


def read_from_corpus(en_file, ko_file):
    kkma = Kkma()
    en_lines = []
    ko_lines = []
    
    print('reading en file')
    with open(en_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            eng_line = line[:-1] #exclude the last '\n'
            # add a space in front of punctuation 
            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            # remove duplicated spaces
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            # split into words
            en_lines.append(eng_line.split())
           
    print('reading ko file')
    with open(ko_file, 'r', encoding='utf-8') as f:
        text=f.read()
        text=text.split('\n')
    #with open(ko_file, 'rb') as f:
    #    for line in tqdm(f):
    for line in tqdm(text):
        # kinlpy.Kkma does morphological analysis.
        words = [tup[0] for tup in kkma.pos(line)]
        ko_lines.append(words)
    
    return en_lines, ko_lines


def unk_words(eng_lines, ko_lines, eng_vocab, ko_vocab, threshold=5):
    for eng_line in eng_lines:
        for i in range(len(eng_line)):
            if eng_vocab[eng_line[i]] <= threshold:
                eng_line[i] = "*UNK*"

    for ko_line in ko_lines:
        for i in range(len(ko_line)):
            if ko_vocab[ko_line[i]] <= threshold:
                ko_line[i] = "*UNK*"


def preprocess_vanilla(eng_file, ko_file, threshold=5):
    """
    preprocess_vanilla unks the corpus and returns two lists of lists of words.

    :param corpus_file: file of the corpus
    :param threshold: threshold count to UNK
    """
    print('reading files')
    eng_lines, ko_lines = read_from_corpus(eng_file, ko_file)
    print('unk words')
    eng_vocab, ko_vocab = count_vocabs(eng_lines, ko_lines)
    unk_words(eng_lines, ko_lines, eng_vocab, ko_vocab, threshold)
    return eng_lines, ko_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("en_file")
    parser.add_argument("ko_file")
    parser.add_argument("en_savefilename")
    parser.add_argument("ko_savefilename")
    args = parser.parse_args()
    
    en_lines, ko_lines = preprocess_vanilla(args.en_file, args.ko_file)
    with open(args.en_savefilename, "w") as f:
        for line in en_lines:
            f.write(" ".join(line)+" \n")

    with open(args.ko_savefilename, "w") as f:
        for line in ko_lines:
            f.write(" ".join(line)+" \n")
