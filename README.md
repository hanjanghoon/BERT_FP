Fine-grained Post-training for Multi-turn Response Selection <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================

Implements the model described in the following paper [Fine-grained Post-training for Improving Retrieval-based Dialogue Systems] in NAACL-2021.

```
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
python -u Fine-tuing/Response_selection.py --task ubuntu --is_training
python -u Fine-tuing/Response_selection.py --task douban --is_training
python -u Fine-tuing/Response_selection.py --task e_commerce --is_training
```
###### Testing
```shell
python -u Fine-tuing/Response_selection.py --task ubuntu
python -u Fine-tuing/Response_selection.py --task douban 
python -u Fine-tuing/Response_selection.py --task e_commerce
```


Training Response Selection Models
--------

### Model Arguments

##### Fine-grained post-training

| task_name  | data_dir              |  checkpoint_path                  |
| ---------- | --------------------- |  ------------------------------------- |
| ubuntu     | ubuntu_data/ubuntu_post_train.pkl         | FPT/PT_checkpoint/ubuntu/bert.pt    |
| douban     | douban_data/douban_post_train.pkl         | FPT/PT_checkpoint/douban/bert.pt    |
| e-commerce | e_commerce_data/e_commerce_post_train.pkl | FPT/PT_checkpoint/e_commerce/bert.pt|

##### Fine-tuning

| task_name  | data_dir              |  checkpoint_path                  |
| ---------- | --------------------- |  ------------------------------------- |
| ubuntu     | ubuntu_data/ubuntu_dataset_1M.pkl         | Fine-tuning/FT_checkpoint/ubuntu.0.pt    |
| douban     | douban_data/douban_dataset_1M.pkl         | Fine-tuning/FT_checkpoint/douban.0.pt    |
| e-commerce | e_commerce_data/e_commerce_dataset_1M.pkl | Fine-tuning/FT_checkpoint/e_commerce.0.pt|

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
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[7]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[8]: https://github.com/cooelf/DeepUtteranceAggregation

[6]: https://github.com/ymcui/Chinese-BERT-wwm
[7]: https://github.com/ymcui/Chinese-ELECTRA
[8]:https://drive.google.com/file/d/14jxet4niR7o_kml8Wp77kFh24C9rVAPT/
[9]:https://drive.google.com/file/d/1kPd3HpAAkEACZDUs1WZ_vkNAXVvnnzq7/
[10]:https://drive.google.com/file/d/15k69AtGjwfB81_qP2K7xL2CdcgWvuz0I/
