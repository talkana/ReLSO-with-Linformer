#!/bin/bash

models=("Relso" "RelsoLin_k25", "RelsoLin_k60")
embed_sizes=(20 100)

for model in "${models[@]}"; do
    for embed_size in "${embed_sizes[@]}"; do
        model_folder="TAPE_400epochs/${model}_embed${embed_size}"
        python3 run_optim.py --weights "${model_folder}/model_state.ckpt" --embeddings "${model_folder}/train_embeddings.npy" --dataset TAPE --det_inits --log_dir "optim_logs/${model}_embed${embed_size}"
    done
done
