from comet_ml import Experiment
from preprocess_unsup import TranslationDataset
from model_unsup import Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
"""
accuracy: 0.09776391055642225
command:
python translation_unsup.py -Tts data/train_en_UNK.txt data/train_ko_UNK.txt data/test_en_UNK.txt data/test_ko_UNK.txt
"""

hyperparams = {
    "rnn_size": 128,  # encoder and decoder use the same rnn_size
    "embedding_size": 128,
    "num_epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.005,
    "seq_len":40
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams, 
          to_en_id = 1, to_ko_id = 4):
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
    model = model.train()
    with experiment.train():
        for epoch in range(hyperparams['num_epochs']):
            for batch in tqdm(train_loader):
                with torch.no_grad(): # ...???
                    trans_ko2en, trans_ko2en_lengths = model.translate(batch["ko_enc_input"].to(device),batch["len_ko_enc"],to_en_id)
                    trans_en2ko, trans_en2ko_lengths = model.translate(batch["en_enc_input"].to(device),batch["len_en_enc"],to_ko_id)
                optimizer.zero_grad()
                output_ko2ko=model(batch["ko_enc_input"].to(device), batch["ko_dec_input"].to(device),
                              batch["len_ko_enc"],batch["len_ko_dec"])
                output_en2en=model(batch["en_enc_input"].to(device), batch["en_dec_input"].to(device),
                              batch["len_en_enc"],batch["len_en_dec"])
                output_ko2en=model(trans_ko2en, batch["ko_dec_input"].to(device),
                              trans_ko2en_lengths,batch["len_ko_dec"])
                output_en2ko=model(trans_en2ko, batch["en_dec_input"].to(device),
                              trans_en2ko_lengths,batch["len_en_dec"])                
                loss_auto = (loss_fn(torch.transpose(output_ko2ko, 1,2),batch["ko_dec_output"].to(device))
                            +loss_fn(torch.transpose(output_en2en, 1,2),batch["en_dec_output"].to(device)))
                loss_cd = (loss_fn(torch.transpose(output_ko2en, 1,2),batch["ko_dec_output"].to(device))
                          +loss_fn(torch.transpose(output_en2ko, 1,2),batch["en_dec_output"].to(device)))
                loss = loss_auto+loss_cd
                loss.backward()
                optimizer.step()            
    experiment.end()

def test(model, test_loader, experiment, hyperparams,
         to_en_id = 1):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # Define loss function, total loss, and total word count
    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    #total_loss = 0
    word_count = 0
    correct_count = 0
    model = model.eval()
    with experiment.test():
        with torch.no_grad():
            for batch in tqdm(test_loader):
                outputs,outputs_lengths = model.translate(batch["ko_enc_input"].to(device),batch["len_ko_enc"],to_en_id) # (batch_size, dec_seq_len)
                #outputs,outputs_lengths = model.translate_teacher_forcing(batch["ko_enc_input"].to(device),batch["len_ko_enc"].to(device),
                #                          batch["en_dec_input"].to(device),batch["len_en_dec"].to(device),to_en_id) # (batch_size, dec_seq_len)
                
                # loss is scalar tensor --it's averaged over batch
                #loss = loss_fn(torch.transpose(outputs, 1,2),batch["tgt_output"])
                word_num = torch.sum(batch["len_en_dec"])
                #total_loss+= torch.tensor(loss.item()) * word_num
                word_count+= word_num
                #max_logit, argmax_logit = torch.max(outputs, dim=2) #argmax_logit = (batch_size, dec_seq_len)
                # don't include pad tokens
                is_correct = torch.eq(batch["en_dec_output"].to(device), outputs) # (batch_size, dec_seq_len)
                for is_c, l in zip(is_correct, batch["len_en_dec"]):
                    correct_count+= torch.sum(is_c[:l])
                #perplexity = torch.exp(total_loss/word_count)
                accuracy = correct_count.item() / word_count.item()
                #experiment.log_metric("perplexity", perplexity)
                experiment.log_metric("accuracy", accuracy)
        #print("perplexity:", perplexity)
        print("accuracy:", accuracy)


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
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # Load dataset
    dataset_train = TranslationDataset(args.en_file,args.ko_file, 
                            hyperparams["seq_len"]) 
    dataset_test = TranslationDataset(args.en_test_file,args.ko_test_file, 
                            hyperparams["seq_len"],word2id = dataset_train.word2id_joint) 
    vocab_size = len(dataset_test.word2id_joint)
    dataLoader_train = DataLoader(dataset_train, batch_size=hyperparams["batch_size"], shuffle=True)
    dataLoader_test = DataLoader(dataset_test, batch_size=hyperparams["batch_size"])

    model = Seq2Seq(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        hyperparams["seq_len"],
        3
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model_unsup.pt'))
    if args.train:
        print("running training loop...")
        train(model, dataLoader_train, experiment, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, dataLoader_test, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model_unsup.pt')
