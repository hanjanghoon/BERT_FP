from transformers import BertTokenizer
import pickle
from tqdm import tqdm

def FT_data(file, tokenizer=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [ [sen for sen in a[1:]] for a in data]
    cr_list=[]
    cnt=0
    for s in tqdm(cr):
        s_list=[]
        for sen in s[:-1]:
            if len(sen)==0:
                cnt+=1
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    print(cnt)
    return y, cr_list

def PT_data():
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open('douban_data/train.txt', 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [[sen for sen in a[1:]] for a in data]
    crnew=[]
    for i,crsingle in enumerate(cr):
        if y[i]==1:
            crnew.append(crsingle)
    crnew=crnew
    pickle.dump(crnew, file=open("douban_post_train.pkl", 'wb'))

if __name__ == '__main__':
    #Fine_tuning data constuction
    #including tokenization step
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",do_lower_case=True)
    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)

    train, test, dev = {}, {}, {}
    train['y'], train['cr'] = FT_data('douban_data/train.txt', tokenizer=bert_tokenizer)
    dev['y'], dev['cr'] = FT_data('douban_data/dev.txt', tokenizer=bert_tokenizer)
    test['y'], test['cr']= FT_data('douban_data/test.txt',tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, dev, test
    pickle.dump(dataset, open('douban_data/dataset_1M.pkl', 'wb'))


    #posttraining data construction
    #does not include tokenization step
    PT_data()