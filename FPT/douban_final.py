from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm, trange


import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer,BertConfig
from transformers import BertForPreTraining
from transformers import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import pickle

from torch.utils.data import Dataset
import random
from setproctitle import setproctitle
setproctitle('douban_final')


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0



class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8-sig", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0

            crsets = pickle.load(file=open(corpus_path, 'rb'))
            #crsets=crsets[:100000]#crsets[:50000]+crsets[500000:]
            cnt=0
            lcnt=0
            for crset in tqdm(crsets):
                for line in crset:
                    if len(line) == 0:
                        continue
                    if len(line) < 12:
                        if len(self.tokenizer.tokenize(line)) == 0:
                            # print('\n'+line+'\n')
                            cnt += 1
                            continue
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc),
                              "end": 0
                              }
                    self.sample_to_doc.append(sample)
                    # if len(self.tokenizer.tokenize(line)) == 0:
                    # print("여기")
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

                if (len(doc) != 0):
                    self.all_docs.append(doc)
                else:
                    print("empty")


                if len(doc)<2:
                    print("!!")
                if (len(doc) < 3):
                    #print("problem!")

                    for i in range(len(doc)-1):
                        self.sample_to_doc.pop()

                    self.sample_to_doc[-1]['end']=len(doc)
                    lcnt+=1
                else:
                # remove last added sample because there won't be a subsequent line anymore in the doc
                    self.sample_to_doc.pop()
                    self.sample_to_doc.pop()

                doc = []
            print(cnt,lcnt)

            # if last row in file is not empty
            '''
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()
            '''
            for doc in self.all_docs:
                if len(doc) == 0:
                    print("problem")
            self.num_docs = len(self.all_docs)

    def __len__(self):
        return len(self.sample_to_doc)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        sample = self.sample_to_doc[item]
        #방법에 비해 문장 길이가 짧은경우.
        length=sample['end']

        if length!=0:
            tokens_a=[]
            for i in range(length-1):
                tokens_a+=self.tokenizer.tokenize(self.all_docs[sample["doc_id"]][sample["line"]+i])+[self.tokenizer.eos_token]
            tokens_a.pop()
            self.current_doc = sample["doc_id"]

            rand = random.random()
            if rand > 0.75:

                # 다음문장
                response = self.all_docs[sample["doc_id"]][length - 1]
                is_next_label = 2


            elif rand > 0.5:

                # 네거티브의 반은 그 문장 자체들..즉 context의 문장중 하나임. 그리고 이게 전체 dialog session이라 여긴 괜춘

                rand_idx = random.randint(0, length - 2)

                response = self.all_docs[sample["doc_id"]][rand_idx]

                is_next_label = 1

            else:
                response = self.get_random_line()
                is_next_label = 0

            tokens_b=self.tokenizer.tokenize(response)
           

        else:
            t1, t2, t3, is_next_label = self.random_sent(item)
            
            tokens_a = self.tokenizer.tokenize(t1)+[self.tokenizer.eos_token]+self.tokenizer.tokenize(t2)
            tokens_b = self.tokenizer.tokenize(t3)

        
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

        
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        sample = self.sample_to_doc[index]
        t1 = self.all_docs[sample["doc_id"]][sample["line"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
        self.current_doc = sample["doc_id"]
        rand = random.random()

        if rand > 0.75:
            label = 2
            t3 = self.all_docs[sample["doc_id"]][sample["line"] + 2]
            # used later to avoid random nextSentence from same doc

        elif rand > 0.5:
            samedoc = self.all_docs[sample["doc_id"]]
            linenum = random.randrange(len(samedoc))

            while linenum == sample["line"] + 2:
                linenum = random.randrange(len(samedoc))

            # 같은 dialog session 이지만 다음문장은 아님.
            t3 = samedoc[linenum]
            label = 1

        else:
            # 랜덤이면 0
            t3 = self.get_random_line()
            label = 0

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, t3 ,label


    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line



class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        if token=='[eos]':
            output_label.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    t1_random, t1_label = random_word(tokens_a, tokenizer)
    t2_random, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if len(tokens_b)==0:
        print(example.tokens_b)
    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

 
    input_mask = [1] * len(input_ids)

    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if example.guid < 0:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default="douban_post_train.pkl",
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="out",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=240,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=50,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1.5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        default=True,
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    bertconfig = BertConfig.from_pretrained(args.bert_model)
    model = BertForPreTraining.from_pretrained(args.bert_model, config=bertconfig)

    model.resize_token_embeddings(len(tokenizer))
    model.cls.seq_relationship = nn.Linear(bertconfig.hidden_size, 3)
    #model.bert.load_state_dict(state_dict=torch.load("douban_final/checkpoint28-455552/bert.pt"))
    model.to(device)




    num_train_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_file)
        train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory)
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)


        train_sampler = RandomSampler(train_dataset)


        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=3)
        learning_rate=args.learning_rate
        before = 10
        for epoch in trange(1, int(args.num_train_epochs) + 1, desc="Epoch"):

            tr_loss = 0

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",position=0)):
                with torch.no_grad():
                    batch = (item.cuda(device=device) for item in batch)
                input_ids, input_mask, segment_ids,lm_label_ids, is_next = batch
                model.train()
                optimizer.zero_grad()
                prediction_scores, seq_relationship_score = model(input_ids=input_ids,attention_mask= input_mask, token_type_ids=segment_ids)
                #logits = torch.sigmoid(output[0].squeeze())
                if lm_label_ids is not None and is_next is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size),
                                              lm_label_ids.view(-1))
                    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 3), is_next.view(-1))
                    total_loss = masked_lm_loss + next_sentence_loss

                model.zero_grad()
                loss = total_loss
                if step%100==0:
                    print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(step, loss.item(),args.train_batch_size) )
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                else:
                    loss.backward()
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    if global_step / num_train_steps < args.warmup_proportion:
                        lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            averloss=tr_loss/step
            print("epoch: %d\taverageloss: %f\tstep: %d "%(epoch,averloss,step))
            print("current learning_rate: ", learning_rate)
            if global_step/num_train_steps > args.warmup_proportion and averloss > before - 0.01:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                    learning_rate = param_group['lr']
                print("Decay learning rate to: ", learning_rate)

            before=averloss

            if True:
                # Save a trained model
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                checkpoint_prefix = 'checkpoint' + str(epoch+28)
                output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_dir1 = output_dir + '/bert.pt'
                torch.save(model.bert.state_dict(), output_dir1)



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > 2*len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_model(model, path):
    model.load_state_dict(state_dict=torch.load(path))
    if torch.cuda.is_available(): model.cuda()

if __name__ == "__main__":
    main()