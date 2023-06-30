import os
from argparse import ArgumentParser
from glob import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_embeddings(models_path, main_save_dir, plot_3D=False):
    for model_path in glob(models_path):
        #Plot for one model
        model_label = os.path.split(model_path)[-1]
        embed_n_fit = {name:{"embeddings":None, "fitness":None, "coords":None} for name in ["train", "valid", "test"]}
        #get data
        for path in glob(model_path+"/**/*.npy", recursive=True):
            label = os.path.split(path)[-1][:-4].split("_")
            if label[1]=="embeddings":
                embed_n_fit[label[0]][label[1]] = np.load(path)
                embed_n_fit[label[0]]["coords"] = PCA(n_components=2).fit_transform(embed_n_fit[label[0]][label[1]])
            if label[1]=="fitness":
                embed_n_fit[label[0]][label[1]] = np.load(path)
        # ploting in 2D
        fig, ax = plt.subplots(1, 3, figsize=(15,5), squeeze=False, constrained_layout = True)
        for i, dataset_type in enumerate(embed_n_fit):
            embeddings = embed_n_fit[dataset_type]["embeddings"]
            fitness = embed_n_fit[dataset_type]["fitness"]
            emb_coords = embed_n_fit[dataset_type]["coords"]
            plot = ax[0, i].scatter(emb_coords[:,0], emb_coords[:,1], c=fitness, s=3)
            ax[0, i].set_xlabel('PCA 1')
            ax[0, i].set_ylabel('PCA 2')
            ax[0, i].set_title(f'{cl_args.dataset} {dataset_type}', fontsize=15)
        fig.colorbar(plot, ax = ax[0,2])
        save_dir = main_save_dir + f"/{model_label}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + f"Original_embeddings_{model_label}.png")
        plt.close()

        if plot_3D:
            #plotting in 3D
            fig = plt.figure(figsize=(15,5), constrained_layout = True)
            for i, dataset_type in enumerate(embed_n_fit):
                fitness = embed_n_fit[dataset_type]["fitness"]
                emb_coords = embed_n_fit[dataset_type]["coords"]
                ax = fig.add_subplot(1,3,i+1, projection='3d')
                ax.plot_trisurf(emb_coords[:,0],
                                emb_coords[:,1],
                                fitness,
                                cmap='viridis', edgecolor='none')
                ax.set_title(f'{cl_args.dataset} {dataset_type}', fontsize=15)
            # save figure
            save_dir = main_save_dir + f"/{model_label}/"
            plt.savefig(save_dir + f"Landscape_3D_{model_label}.png")
            plt.close()

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument("--dataset", required=True, type=str)

    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--save_dir", default=None, type=str)

    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    ####################################################################################################
    # #plot embeddings
    if cl_args.model_dir:
        models_path = cl_args.model_dir
    else:
        models_path = f"train_logs/relso/{cl_args.dataset}/*"
    if cl_args.save_dir:
        main_save_dir = cl_args.save_dir
    else:
        main_save_path = f"./figures/{cl_args.dataset}/*"

    plot_embeddings(models_path, main_save_dir, plot_3D=False)
    ####################################################################################################



    




