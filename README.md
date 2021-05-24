Fine-grained Post-training for Multi-turn Response Selection <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================

Implements the model described in the following paper [Fine-grained Post-training for Improving Retrieval-based Dialogue Systems] in NAACL-2021.

```
```
This code is reimplemented as a fork of [huggingface/transformers][2].

![alt text](ums_overview.png)

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

### Pre- and Post-trained Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][3]
- [fine-tuned checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][4]


### Data pkls for Fine-tuning (Response Selection)
We used reconstructed following datasets
- [fine-grained post-training dataset and fine-tuning dataset for 3 benchmarks (ubuntu, douban, e-commerce)][5]


Original version for each dataset is availble in [Ubuntu Corpus V1][3], [Douban Corpus][4], and [E-Commerce Corpus][5], respectively.



Fine-grained Post-Training
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
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://github.com/ymcui/Chinese-BERT-wwm
[7]: https://github.com/ymcui/Chinese-ELECTRA
[8]:https://drive.google.com/file/d/14jxet4niR7o_kml8Wp77kFh24C9rVAPT/
[9]:https://drive.google.com/file/d/1kPd3HpAAkEACZDUs1WZ_vkNAXVvnnzq7/
[10]:https://drive.google.com/file/d/15k69AtGjwfB81_qP2K7xL2CdcgWvuz0I/
