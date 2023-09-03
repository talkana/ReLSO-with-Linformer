
<div align="center">    
 
# LinReLSO: A Transformer-based Model for Latent Space Optimization and Generation of Proteins with Linformer attention
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2201.09948)
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2006.04768)
-->

[![Paper](https://img.shields.io/badge/arxiv-2201.09948-B31B1B.svg)](https://arxiv.org/abs/2201.09948)

[![Paper](https://img.shields.io/badge/arxiv-2006.04768-B31B1B.svg)](https://arxiv.org/abs/2006.04768)

[![DOI](https://zenodo.org/badge/436740631.svg)](https://zenodo.org/badge/latestdoi/436740631)
  
</div>

Improved Fitness Optimization Landscapes
for Sequence Design
- [Description](#Description)
- [Citation](#citation)
- [How to run   ](#how-to-run)
- [Training models](training-models)
- [Original data source](#Original-data-sources)


## Description
---
 The advancement of robust natural language models has increased the ability to learn meaningful representations of protein sequences. Deep transformer-based autoencoders such as [ReLSO](https://arxiv.org/abs/2201.09948) can be trained to jointly generate sequences as well as predict their fitness due to their highly structured latent space. However training and deploying this model can be costly due to its use of the standard self-attention mechanism. To address this, we propose LinReLSO, a model that incorporates the ReLSO architecture with [Linformer](https://arxiv.org/abs/2006.04768) self-attention, and we evaluate its performance in comparison to the original architecture. Our findings demonstrate that LinReLSO not only consumes less resources and speeds up computations but also surpasses the original model in terms of both reconstruction and prediction accuracy. For detailed results and insights, please refer to the final_paper.pdf file.

## Citation

This repository is based on:

* Egbert Castro, Abhinav Godavarthi, Julian Rubinfien, Kevin B. Givechian, Dhananjay Bhaskar, and Smita Krishnaswamy. 2022. ReLSO: A Transformer-based
Model for Latent Space Optimization and Generation of Proteins. [ArXiv:2201.09948](https://arxiv.org/abs/2201.09948)
* [ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers](https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers)
* Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-Attention with Linear Complexity. [ArXiv:2006.04768](https://arxiv.org/abs/2006.04768)

## How to run   
---

First, install dependencies   
```bash
# clone project   
git clone https://github.com/talkana/ReLSO-with-Linformer.git

# install requirements

# with conda
conda env create -f relso_env.yml

# with pip
pip install -r requirements.txt
 ```   

## Usage

### Training models
 
 ```bash
# run training script
python train_relso.py  --data TAPE
```
---
*note: if arg option is not relevant to current model selection, it will not be used. See init method of each model to see what's used.

### available dataset args:

        gifford, GB1_WU, GFP, TAPE


### Running optimization algorithms 
 
 ```bash
python run_optim.py --weights <path to ckpt file>/model_state.ckpt --embeddings  <path to embeddings file>train_embeddings.npy --dataset gifford
```
---

## Original data sources

- GIFFORD: https://github.com/gifford-lab/antibody-2019/tree/master/data/training%20data
- GB1: https://elifesciences.org/articles/16965#data
- GFP: https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154
- TAPE: https://github.com/songlab-cal/tape#data

