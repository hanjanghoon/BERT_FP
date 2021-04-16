import time
import argparse
import pickle
import torch
from torch.utils.data import TensorDataset
import os
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from NeuralNetwork_base import NeuralNetwork
from tqdm import tqdm
import random

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import Dataset




class Process:
    def __init__(self):

        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file


        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0

        # load samples into memory
        data = open('ubuntu_data/train.txt', 'r').readlines()
        data = [sent.split('\n')[0].split('\t') for sent in data]
        y = [int(a[0]) for a in data]
        cr = [[sen for sen in a[1:]] for a in data]
        crnew = []
        for i, crsingle in enumerate(cr):
            if y[i] == 1:
                crnew.append(crsingle)
        crsets = crnew

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.sample_to_doc = []
        self.all_docs = []
        doc = []
        # crsets=crsets[:50000]#crsets[:50000]+crsets[500000:]
        cnt = 0
        print("로드끝")

        for crset in tqdm(crsets):
            tempcnt = 0
            crset=crset[:-1]
            for i, line in enumerate(crset):
                if len(line) == 0:
                    tempcnt += 1
                    continue
                if len(line) < 10:
                    if len(self.tokenizer.tokenize(line)) == 0:
                        # print('\n'+line+'\n')
                        cnt += 1
                        tempcnt += 1
                        continue

                sample = {"doc_id": len(self.all_docs),
                          "line": len(doc)}
                self.sample_to_doc.append(sample)
                doc.append(line)

            if (len(doc) != 0):
                self.all_docs.append(doc)
            else:
                print("empty")
            doc = []
        print(cnt)
        for doc in self.all_docs:
            if len(doc) == 0:
                print("problem")
        self.num_docs = len(self.all_docs)


    def __len__(self):
        return len(self.sample_to_doc)


    def random_sent(self, index):
        sample = self.sample_to_doc[index]
        self.current_doc = sample["doc_id"]

        context_len=sample["line"]
        if context_len==0:
            return -1,-1,-1
        context=[]
        for i in range(context_len):
            utterance=self.all_docs[sample["doc_id"]][i]
            context+=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance))+[self.tokenizer.eos_token_id]

        utterance = self.all_docs[sample["doc_id"]][context_len]
        response= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance))

        utterance=self.get_random_line()
        negative = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance))

        assert len(context) > 0
        assert len(response) > 0
        assert len(negative) > 0
        return context,response,negative


    def get_random_line(self):
        for _ in range(10):

            rand_doc_idx = random.randint(0, len(self.all_docs)-1)
            rand_doc = self.all_docs[rand_doc_idx]
            line = rand_doc[random.randrange(len(rand_doc))]
            if self.current_random_doc != self.current_doc:
                break
        return line


    def makedata(self,item):

        context, response, negative= self.random_sent(item)
        if context==-1:
            return -1,-1

        truecrslist=context+[self.tokenizer.sep_token_id]+response
        falsecrlist=context+[self.tokenizer.sep_token_id]+negative
        return truecrslist,falsecrlist

def data_augmentation():
    object=Process()
    newdata={}
    newdata['y']=[]
    newdata['cr']=[]
    for i in tqdm(range(object.__len__())):
        truecrlist,falsecrlist=object.makedata(i)
        if truecrlist==-1:
            continue
        newdata['y'].append(1)
        newdata['cr'].append(truecrlist)
        newdata['y'].append(0)
        newdata['cr'].append(falsecrlist)

    pickle.dump(newdata, file=open("augmentation_train.pkl", 'wb'))


if __name__ == '__main__':
    data_augmentation()




