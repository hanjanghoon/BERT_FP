import time
import argparse
import pickle
import torch
from torch.utils.data import TensorDataset
import os
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle

setproctitle('BERT_FP')

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FT_data={
    'ubuntu': '../ubuntu_data/ubuntu_dataset_1M.pkl',
    'douban': '../douban_data/douban_dataset_1M.pkl',
    'e_commerce': '../e_commerce_data/e_commerce_dataset_1M.pkl'
}

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=1,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="sopscr.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--cache_dir", default="bert_cache", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
args = parser.parse_args()
args.save_path += args.task + '.' + "0.pt"
args.score_file_path = args.task+ '_' +  args.score_file_path
# load bert


print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training==True:
        train_model(train,dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




