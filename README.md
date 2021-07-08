Fine-grained Post-training for Multi-turn Response Selection <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-post-training-for-improving/conversational-response-selection-on-ubuntu-1)](https://paperswithcode.com/sota/conversational-response-selection-on-ubuntu-1?p=fine-grained-post-training-for-improving)


Implements the model described in the following paper [Fine-grained Post-training for Improving Retrieval-based Dialogue Systems](https://www.aclweb.org/anthology/2021.naacl-main.122/) in NAACL-2021.

```
@inproceedings{han-etal-2021-fine,
title = "Fine-grained Post-training for Improving Retrieval-based Dialogue Systems",
author = "Han, Janghoon  and Hong, Taesuk  and Kim, Byoungjae  and Ko, Youngjoong  and Seo, Jungyun",
booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
month = jun, year = "2021", address = "Online", publisher = "Association for Computational Linguistics", url = "https://www.aclweb.org/anthology/2021.naacl-main.122", pages = "1549--1558",
}
```
This code is reimplemented as a fork of [huggingface/transformers][2].

![alt text](https://user-images.githubusercontent.com/32722198/119294465-79847080-bc8f-11eb-80b1-1dfafb0e1b6c.png)

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][3]
- [fine-tuned checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][4]


### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for 3 benchmarks (ubuntu, douban, e-commerce)][5]


Original version for each dataset is availble in [Ubuntu Corpus V1][6], [Douban Corpus][7], and [E-Commerce Corpus][8], respectively.


Fine-grained Post-Training
--------

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
python -u FPT/douban_final.py --num_train_epochs 27
python -u FPT/e_commmerce_final.py --num_train_epochs 34
```

### Fine-tuning Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
python -u Fine-Tuning/Response_selection.py --task douban --is_training
python -u Fine-Tuning/Response_selection.py --task e_commerce --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
python -u Fine-Tuning/Response_selection.py --task douban 
python -u Fine-Tuning/Response_selection.py --task e_commerce
```


Training Response Selection Models
--------

### Model Arguments

##### Fine-grained post-training

| task_name  | data_dir                                  |  checkpoint_path                    |
| ---------- | ---------------------                     |  -----------------------------------|
| ubuntu     | ubuntu_data/ubuntu_post_train.pkl         | FPT/PT_checkpoint/ubuntu/bert.pt    |
| douban     | douban_data/douban_post_train.pkl         | FPT/PT_checkpoint/douban/bert.pt    |
| e-commerce | e_commerce_data/e_commerce_post_train.pkl | FPT/PT_checkpoint/e_commerce/bert.pt|

##### Fine-tuning

| task_name     | data_dir                                  |  checkpoint_path                         |
| ----------    | ---------------------                     |  ----------------------------------------|
| ubuntu        | ubuntu_data/ubuntu_dataset_1M.pkl         | Fine-Tuning/FT_checkpoint/ubuntu.0.pt    |
| douban        | douban_data/douban_dataset_1M.pkl         | Fine-Tuning/FT_checkpoint/douban.0.pt    |
| e-commerce    | e_commerce_data/e_commerce_dataset_1M.pkl | Fine-Tuning/FT_checkpoint/e_commerce.0.pt|



Performance
----------

We provide model checkpoints of BERT_FP, which obtained new state-of-the-art, for each dataset.

| Ubuntu         | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP]      | 0.911 | 0.962 | 0.994 |

| Douban         | MAP   | MRR   | P@1   | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| [BERT_FP]      | 0.644 | 0.680 | 0.512 | 0.324 | 0.542 | 0.870 |

| E-Commerce     | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP]      | 0.870 | 0.956 | 0.993 |



[1]: https://github.com/huggingface/transformers
[2]: https://github.com/taesunwhang/BERT-ResSel
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[7]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[8]: https://github.com/cooelf/DeepUtteranceAggregation
