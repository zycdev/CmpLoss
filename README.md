# CmpLoss

This repository contains the code and processed data for the
paper [Cross-Model Comparative Loss for Enhancing Neuronal Utility in Language Understanding](https://arxiv.org/pdf/2301.03765.pdf).

Comparative loss is a simple task-agnostic loss function to improve neuronal utility without additional human
supervision.
It is essentially a pairwise ranking loss based on the comparison principle between the full model and its ablated
models, with the expectation that the less ablation there is, the smaller the task-specific loss.
It is theoretically applicable to all dropout-compatible models and tasks whose inputs contain irrelevant content.

Our experiments are conducted in the following environment with V100 (32 GB) GPUs.

```shell
conda create -n pt11 python=3.8
conda activate pt11
conda install -c conda-forge jupyterlab=3.4.5 tensorboard=2.10.0 ipywidgets=8.0.2
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge scikit-learn=1.1.1 scipy=1.8.1
conda install -c huggingface -c conda-forge tokenizers=0.12.1 datasets=2.1.0 transformers=4.19.2
pip install python-Levenshtein==0.20.8 matplotlib==3.6.2

cd data/hotpot/ && tar -jxvf corpus.distractor.tsv.tar.bz2 && cd -
```

The commands to run the experiments on GLUE, SQuAD, and HotpotQA are placed
in [scripts](https://github.com/zycdev/CmpLoss/tree/main/scripts).
The code and models for the experiments on MS MARCO can be found at [LoL](https://github.com/zycdev/LoL).
If you want to train your models on other tasks with comparative loss, you can refer to `CmpQA`
in [modeling.py](https://github.com/zycdev/CmpLoss/blob/main/modeling.py#197)
and [run_hotpot.py](https://github.com/zycdev/CmpLoss/blob/main/run_hotpot.py).

If you use comparative loss in your work, please consider citing our paper:

```
@misc{zhu2023cmp,
  title = {Cross-{{Model Comparative Loss}} for {{Enhancing Neuronal Utility}} in {{Language Understanding}}},
  author = {Zhu, Yunchang and Pang, Liang and Wu, Kangxi and Lan, Yanyan and Shen, Huawei and Cheng, Xueqi},
  year = {2023},
  month = jan,
  eprint = {2301.03765},
  eprinttype = {arxiv},
  url = {http://arxiv.org/abs/2301.03765}
}
```