UMS for Multi-turn Response Selection <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/do-response-selection-models-really-know-what/conversational-response-selection-on-ubuntu-1)](https://paperswithcode.com/sota/conversational-response-selection-on-ubuntu-1?p=do-response-selection-models-really-know-what)

Implements the model described in the following paper [Do Response Selection Models Really Know What's Next? Utterance Manipulation Strategies for Multi-turn Response Selection](https://arxiv.org/abs/2009.04703).

```
@inproceedings{whang2021ums,
  title={Do Response Selection Models Really Know What's Next? Utterance Manipulation Strategies for Multi-turn Response Selection},
  author={Whang, Taesun and Lee, Dongyub and Oh, Dongsuk and Lee, Chanhee and Han, Kijong and Lee, Dong-hun and Lee, Saebyeok},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

This code is reimplemented as a fork of [huggingface/transformers][1] and [taesunwhang/BERT-ResSel][1].

![alt text](ums_overview.png)

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.6.0, and provides out of the box support with CUDA 10.1 and CuDNN 7.6.5.

Anaconda / Miniconda is the recommended to set up this codebase.

### Anaconda or Miniconda

Clone this repository and create an environment:

```shell
git clone https://www.github.com/taesunwhang/UMS-ResSel
conda create -n ums_ressel python=3.7

# activate the environment and install all dependencies
conda activate ums_ressel
cd UMS-ResSel

# https://pytorch.org
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Pre- and Post-trained Checkpoints

We provide following pre- and post-trained checkpoints. 

- [bert-base (english)][1], [bert-base-wwm (chinese)][6]
- bert-post (ubuntu, douban, e-commerce)
- [electra-base (english)][1], [electra-base (chinese)][7]
- electra-post (ubuntu, douban, e-commerce)

```shell
sh scripts/download_pretrained_checkpoints.sh
```

### Data pkls for Fine-tuning (Response Selection)

Original version for each dataset is availble in [Ubuntu Corpus V1][3], [Douban Corpus][4], and [E-Commerce Corpus][5], respectively.

``` shell
sh scripts/download_datasets.sh
```



Domain-specific Post-Training
--------

### Post-training Creation

##### Data for post-training BERT

```shell
#Ubuntu Corpus V1
sh scripts/create_bert_post_data_creation_ubuntu.sh
#Douban Corpus
sh scripts/create_bert_post_data_creation_douban.sh
#E-commerce Corpus
sh scripts/create_bert_post_data_creation_e-commerce.sh
```

##### Data for post-training ELECTRA 

```shell
sh scripts/download_electra_post_training_pkl.sh
```



### Post-training Examples

##### BERT+ (e.g., Ubuntu Corpus V1)

```shell
python3 main.py --model bert_post_training --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-base-uncased --bert_checkpoint_path bert-base-uncased-pytorch_model.bin --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir --training_type post_training
```

##### ELECTRA+ (e.g., Douban Corpus)

```shell
python3 main.py --model electra_post_training --task_name douban --data_dir data/electra_post_training --bert_pretrained electra-base-chinese --bert_checkpoint_path electra-base-chinese-pytorch_model.bin --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir --training_type post_training
```



Training Response Selection Models
--------

### Model Arguments

##### BERT-Base

| task_name              | data_dir                         | bert_pretrained       | bert_checkpoint_path                |
| ---------------------- | -------------------------------- | --------------------- | ----------------------------------- |
| ubuntu                 | data/ubuntu_corpus_v1            | bert-base-uncased     | bert-base-uncased-pytorch_model.bin |
| douban<br />e-commerce | data/douban<br />data/e-commerce | bert-base-wwm-chinese | bert-base-wwm-chinese_model.bin     |

##### BERT-Post

| task_name  | data_dir              | bert_pretrained     | bert_checkpoint_path                  |
| ---------- | --------------------- | ------------------- | ------------------------------------- |
| ubuntu     | data/ubuntu_corpus_v1 | bert-post-uncased   | bert-post-uncased-pytorch_model.pth   |
| douban     | data/douban           | bert-post-douban    | bert-post-douban-pytorch_model.pth    |
| e-commerce | data/e-commerce       | bert-post-ecommerce | bert-post-ecommerce-pytorch_model.pth |

##### ELECTRA-Base

| task_name              | data_dir                         | bert_pretrained      | bert_checkpoint_path                   |
| ---------------------- | -------------------------------- | -------------------- | -------------------------------------- |
| ubuntu                 | data/ubuntu_corpus_v1            | electra-base         | electra-base-pytorch_model.bin         |
| douban<br />e-commerce | data/douban<br />data/e-commerce | electra-base-chinese | electra-base-chinese-pytorch_model.bin |

##### ELECTRA-Post

| task_name  | data_dir              | bert_pretrained        | bert_checkpoint_path                     |
| ---------- | --------------------- | ---------------------- | ---------------------------------------- |
| ubuntu     | data/ubuntu_corpus_v1 | electra-post           | electra-post-pytorch_model.pth           |
| douban     | data/douban           | electra-post-douban    | electra-post-douban-pytorch_model.pth    |
| e-commerce | data/e-commerce       | electra-post-ecommerce | electra-post-ecommerce-pytorch_model.pth |



### Fine-tuning Examples

##### BERT+ (e.g., Ubuntu Corpus V1)

```shell
python3 main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir
```

##### UMS BERT+ (e.g., Douban Corpus)

```shell
python3 main.py --model bert_post --task_name douban --data_dir data/douban --bert_pretrained bert-post-douban --bert_checkpoint_path bert-post-douban-pytorch_model.pth --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir --multi_task_type "ins,del,srch"
```

##### UMS ELECTRA (e.g., E-Commerce)

```shell
python3 main.py --model electra_base --task_name e-commerce --data_dir data/e-commerce --bert_pretrained electra-base-chinese --bert_checkpoint_path electra-base-chinese-pytorch_model.bin --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir --multi_task_type "ins,del,srch"
```

Evaluation
----------
To evaluate the model, set `--evaluate` to `/path/to/checkpoints` 

##### UMS BERT+ (e.g., Ubuntu Corpus V1)

```shell
python3 main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --gpu_ids "0" --root_dir /path/to/root_dir --evaluate /path/to/checkpoints --multi_task_type "ins,del,srch"
```



Performance
----------

We provide model checkpoints of UMS-BERT+, which obtained new state-of-the-art, for each dataset.

| Ubuntu         | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [UMS-BERT+][8] | 0.875 | 0.942 | 0.988 |

| Douban         | MAP   | MRR   | P@1   | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| [UMS-BERT+][9] | 0.625 | 0.664 | 0.499 | 0.318 | 0.482 | 0.858 |

| E-Commerce      | R@1   | R@2   | R@5   |
| --------------- | ----- | ----- | ----- |
| [UMS-BERT+][10] | 0.762 | 0.905 | 0.986 |



[1]: https://github.com/huggingface/transformers
[2]: https://github.com/taesunwhang/BERT-ResSel
[3]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[4]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[5]: https://github.com/cooelf/DeepUtteranceAggregation
[6]: https://github.com/ymcui/Chinese-BERT-wwm
[7]: https://github.com/ymcui/Chinese-ELECTRA
[8]:https://drive.google.com/file/d/14jxet4niR7o_kml8Wp77kFh24C9rVAPT/
[9]:https://drive.google.com/file/d/1kPd3HpAAkEACZDUs1WZ_vkNAXVvnnzq7/
[10]:https://drive.google.com/file/d/15k69AtGjwfB81_qP2K7xL2CdcgWvuz0I/
