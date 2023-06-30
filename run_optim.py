import datetime
import os
from argparse import ArgumentParser

import numpy as np
from pytorch_lightning.loggers import WandbLogger

import relso.data as hdata
from relso.nn.models import relso1
from relso.optim import optim_algs
from relso.optim import utils as opt_utils
from relso.utils import eval_utils

if __name__ == '__main__':

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument('--weights', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--n_steps', default=200, type=int)
    parser.add_argument('--log_dir', default='optim_logs/', type=str)
    parser.add_argument('--log_iter', default=None, type=int)
    parser.add_argument('--project_name', default='relso-optim', type=str)
    parser.add_argument('--det_inits', default=False, action='store_true')
    parser.add_argument('--alpha', required=False, type=float)
    parser.add_argument('--delta', required=False, default='adaptive', type=str)
    parser.add_argument('--k', required=False, default=5, type=int,
                        help="the value of k that influences the adaptive delta")

    cl_args = parser.parse_args()

    # logging
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")

    if cl_args.log_dir:
        save_dir = f"{cl_args.log_dir}/"

    else:
        save_dir = f'optim_logs/relso1/{cl_args.dataset}/ns{cl_args.n_steps}/{date_suffix}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb_logger = WandbLogger(name=f'run_relso1_{cl_args.dataset}',
                               project=cl_args.project_name,
                               log_model=False,
                               save_dir=save_dir,
                               offline=False)

    wandb_logger.log_hyperparams(cl_args.__dict__)
    wandb_logger.experiment.log({"logging timestamp": date_suffix})

    # load model
    model = relso1.load_from_checkpoint(cl_args.weights)
    model.eval()

    # load dataset
    proto_data = hdata.str2data(cl_args.dataset)
    data = proto_data(dataset=cl_args.dataset,
                      task='recon',
                      batch_size=100)

    model.seq_len = data.seq_len
    *_, train_targs = data.train_split.tensors
    train_targs = train_targs.numpy()

    # load embeddings
    embeddings = np.load(cl_args.embeddings)
    print(f'embeddings loaded with shape: {embeddings.shape}')

    # randomly initialize point
    n_steps = cl_args.n_steps
    num_inits = 10
    num_optim_algs = 1
    optim_algo_names = ['Gradient Ascent']

    optim_embedding_traj_array = np.zeros((num_inits, num_optim_algs, n_steps, embeddings.shape[-1]))
    optim_fitness_traj_array = np.zeros((num_inits, num_optim_algs, n_steps))
    optim_sequences = np.zeros((num_inits, 22, data.seq_len))

    if cl_args.det_inits:
        print('deterministic seeds selected!')
        seed_vals = np.linspace(0,len(embeddings)-1, num_inits)
    else:
        print('random seeds selected!')
        seed_vals = np.random.choice(np.arange(len(embeddings)), num_inits)


    if cl_args.delta == 'adaptive':
        print('adaptive delta selected - computing delta based off pairwise distances')
        cl_args.delta = eval_utils.get_avg_distance(embeddings=embeddings, k=cl_args.k)

    else:
        cl_args.delta = float(cl_args.delta)

    for run_indx, init_indx in enumerate(seed_vals):

        init_indx = int(init_indx)
        print(f'\nrunning initialization {init_indx}/{num_inits}\n')

        init_point = embeddings[init_indx].copy()

        # Gradient Ascent
        print("\n")
        embedding_array_ga, fitness_array_ga, out_seq_array_ga = optim_algs.grad_ascent(initial_embedding=init_point.copy(),
                                                                      model=model,
                                                                      N_steps=n_steps,
                                                                      lr=0.1)
        last_seq = out_seq_array_ga[-1].squeeze()  # shape 22 * seqlen
        optim_sequences[run_indx] = last_seq
        print(f'shape of output embedding array: {embedding_array_ga.shape}')
        print('init embed from output: {}'.format(embedding_array_ga[0][:10]))

        run_optim_embeddings = [embedding_array_ga]
        run_optim_fitnesses = [fitness_array_ga]

        for alg_indx, (embed, fit) in enumerate(zip(run_optim_embeddings, run_optim_fitnesses)):
            optim_embedding_traj_array[run_indx, alg_indx] = embed
            optim_fitness_traj_array[run_indx, alg_indx] = fit

    # save embeddings
    print("saving embeddings")
    np.save(save_dir + 'optimization_embeddings.npy', optim_embedding_traj_array)
    np.save(save_dir + 'optimization_fitnesses.npy', optim_fitness_traj_array)
    np.save(save_dir + 'optimal_sequences.npy', optim_sequences)

    # max fitnesss array shape: num_algos x num_runs
    print("logging max fitness values")
    max_fitness_array = optim_fitness_traj_array[:, :, -1]  # n_init x n_algo array
    opt_utils.plot_boxplot(max_fitness_array, optim_algo_names,
                           wandb_logger=wandb_logger,
                           save_path=save_dir + f'max_fitness_boxplot.png')

    # log max fitness values
    # optim_fitness_traj_array shape: n_inits x n_algos x n_steps
    per_algo_fitness_values = optim_fitness_traj_array.transpose(1, 0, 2).reshape(len(optim_algo_names), -1)
    print(f'len of optim_algo_names: {len(optim_algo_names)}')
    print(f'len of max_fitness_array: {len(per_algo_fitness_values)}')

    for name, fit_vals in zip(optim_algo_names, per_algo_fitness_values):
        max_fit_i = fit_vals.max()
        wandb_logger.experiment.log({f'Max Fitness for {name} Runs': max_fit_i})

    endpoint_embed_array = optim_embedding_traj_array.transpose(1, 0, 2, 3)
    # shape will now be num_algo x n_steps x n_inits x embed_dim

    for indx, (name, embeds) in enumerate(zip(optim_algo_names, endpoint_embed_array)):
        print(f'shape of embeddings: {embeddings.shape}')
        print(f'shape of embeds: {embeds.shape}')

        opt_utils.plot_embedding_end_points(embeddings, train_targs, embeds, algo_name=name,
                                            wandb_logger=wandb_logger,
                                            save_path=save_dir + f'max_fitness_PCA_end_points_{indx}.png')

    emb_pca_coords = opt_utils.plot_embedding(embeddings, train_targs,
                                              wandb_logger=wandb_logger,
                                              save_path=save_dir + 'original_fitness_lanscape_pca.png')
