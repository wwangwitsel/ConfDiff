# ConfDiff

This repository is the official implementation of the paper "Binary Classification with Confidence Difference" and technical details of this approach can be found in the paper.


## Requirements:
- Python 3.6.13
- numpy 1.19.2
- Pytorch 1.7.1
- torchvision 0.8.2
- pandas 1.1.5
- scipy 1.5.4


## Arguments:
- mo: model
- ds: data set
- uci: uci dataset or not
- lr: learning rate
- wd: weight decay
- gpu: the gpu index
- ep: training epoch number
- bs: training batch size
- pretrain_bs: batch size for training the probabilistic classifier
- pretrain_ep: epoch number for training the probabilistic classifier
- me: method name
- prior: class prior probability
- n: number of unlabeled data pairs
- run_times: random running times

## Demo:
```
python main.py -mo mlp -ds mnist -uci 0 -lr 1e-3 -wd 1e-5 -gpu 0 -ep 200 -seed 0 -bs 256 -pretrain_bs 256 -pretrain_ep 10 -me ConfDiffABS -prior 0.5 -n 15000 -run_times 5
```

## Citation
```
@inproceedings{wang2023binary,
    author = {Wang, Wei and Feng, Lei and Jiang, Yuchen and Niu, Gang and Zhang, Min-Ling and Sugiyama, Masashi},
    title = {Binary classification with confidence difference},
    booktitle = {Advances in Neural Information Processing Systems 36},
    year = {2023},
    pages = {5936--5960}
}
```
