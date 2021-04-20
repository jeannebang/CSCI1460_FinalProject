from comet_ml import Experiment
from preprocess_sup import TranslationDataset
from model_sup import Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
from konlpy.tag import Kkma
"""
supervised:
perplexity: 38.43387658991687
accuracy: 0.3226825369311069
command:
python translation_sup.py -Tts data/train_en_UNK.txt data/train_ko_UNK.txt data/test_en_UNK.txt data/test_ko_UNK.txt -m "*2en*"
"""

hyperparams = {
    "rnn_size": 128,  # encoder and decoder use the same rnn_size
    "embedding_size": 128,
    "num_epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.001,
    "seq_len":40
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    total_loss=0
    word_count=0
    model = model.train()
    with experiment.train():
        for epoch in range(hyperparams['num_epochs']):
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                outputs=model(batch["src_input"].to(device), batch["tgt_input"].to(device),
                              batch["len_src"],batch["len_tgt"])
                loss = loss_fn(torch.transpose(outputs, 1,2),batch["tgt_output"].to(device))
                loss.backward()
                optimizer.step()            
                word_num = torch.sum(batch["len_tgt"])
                total_loss+= loss.item() * word_num.item()
                word_count+= word_num.item()
                perplexity = np.exp(total_loss/word_count)
                experiment.log_metric("perplexity", perplexity)
    experiment.end()

def test(model, test_loader, experiment, hyperparams):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    word_count = 0
    correct_count = 0
    model.eval()
    with experiment.test():
        with torch.no_grad():
            for batch in tqdm(test_loader):
                outputs=model(batch["src_input"].to(device), batch["tgt_input"].to(device),
                              batch["len_src"],batch["len_tgt"]) # (batch_size, dec_seq_len, output_size)
                # loss is scalar tensor --it's averaged over batch
                loss = loss_fn(torch.transpose(outputs, 1,2),batch["tgt_output"].to(device))
                word_num = torch.sum(batch["len_tgt"])
                total_loss+= loss.item() * word_num.item()
                word_count+= word_num.item()
                max_logit, argmax_logit = torch.max(outputs, dim=2) #argmax_logit = (batch_size, dec_seq_len)
                # don't include pad tokens
                is_correct = torch.eq(batch["tgt_output"].to(device), argmax_logit) # (batch_size, dec_seq_len)
                for is_c, l in zip(is_correct, batch["len_tgt"]):
                    correct_count+= torch.sum(is_c[:l])
                perplexity = np.exp(total_loss/word_count)
                accuracy = correct_count.item() / word_count
                experiment.log_metric("perplexity", perplexity)
                experiment.log_metric("accuracy", accuracy)
        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
    experiment.end()


def interactive(input,word2id,model,tag_token,end_token="*END*", unk_token="*UNK*"):
    if 'en' in tag_token:
        kkma=Kkma()
        input_ko = [tup[0] for tup in kkma.pos(input)]
        input = " ".join(input_ko)
    input_seq = torch.zeros((1,hyperparams["seq_len"]),dtype=int)
    initial_sentence = tag_token +' '+ input +' '+ end_token
    input_token =[]
    for word in initial_sentence:
        if word in word2id:
            input_token.append(word2id[word])
        else:
            input_token.append(word2id[unk_token])
    len_seq = min(len(input_token), hyperparams["seq_len"])
    input_seq[0,:len_seq] = torch.tensor(input_token[:len_seq])

    output_seq = torch.zeros((1,hyperparams["seq_len"]),dtype=int)
    output_seq[0] = word2id[tag_token]

    model = model.eval()
    for i in range(len_seq-1):
        len_output = i+1
        out = model(input_seq.to(device),output_seq.to(device),[len_seq],[len_output])
        next_id = torch.argmax(out[:,i,:],dim=1)
        out[:,i+1] = next_id
        if next_id == word2id[end_token]:
            break

    reverse_vocab = {idx:word for word, idx in word2id.items()}
    out_text = [reverse_vocab[id] for id in out[1:len_output-1]]

    print(" ".join(out_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("en_file")
    parser.add_argument("ko_file")
    parser.add_argument("en_test_file")
    parser.add_argument("ko_test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-m", "--multilingual-tags", nargs="*", default=[None],
                        help="target tags for translation")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # Load dataset
    data_tags = args.multilingual_tags[0]
    dataset_train = TranslationDataset(args.en_file,args.ko_file, 
                            hyperparams["seq_len"],target_token = data_tags) 
    dataset_test = TranslationDataset(args.en_test_file,args.ko_test_file, 
                            hyperparams["seq_len"],word2id = dataset_train.word2id_joint,
                            target_token = data_tags) 
    vocab_size = len(dataset_test.word2id_joint)
    dataLoader_train = DataLoader(dataset_train, batch_size=hyperparams["batch_size"], shuffle=True)
    dataLoader_test = DataLoader(dataset_test, batch_size=hyperparams["batch_size"])

    model = Seq2Seq(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        hyperparams["seq_len"],
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model_sup.pt'))
    if args.train:
        print("running training loop...")
        train(model, dataLoader_train, experiment, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, dataLoader_test, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model_sup_test.pt')
    if args.interactive:
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, dataset_test.word2id_joint, model, data_tags)

