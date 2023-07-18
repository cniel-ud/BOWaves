from argparse import ArgumentParser
from time import perf_counter
from pathlib import Path
import numpy as np

import os
import sys
import matplotlib.pyplot as plt
os.unsetenv('OMP_THREAD_LIMIT')

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means

t_start = perf_counter()

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('experiment', help='Experiment name')
parser.add_argument("--root", help="Path to root folder")
parser.add_argument("--centroid-len", type=int, default=512,
                    help="Centroid length")
parser.add_argument("--window-len", type=int, default=768,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=128, help='Number of clusters')
parser.add_argument('--visualize', default=True, help="Print out codebooks")
parser.add_argument('--visualize_cutoff', type=int, default=5,
                    help="Only centroids with this many occurrences or above are visualized. Set to zero to visualize all")
# TODO - make metric and init type as command line arguments

args = parser.parse_args()
win_len = args.window_len
root = Path(args.root)
data_dir = root.joinpath('data', args.experiment)
data_dir.mkdir(exist_ok=True)


results_dir = root.joinpath('results', args.experiment)
results_dir.mkdir(exist_ok=True)


fpath = list(data_dir.glob('*.npz'))[0]
with np.load(fpath, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

data = np.load(fpath)
# for key in data.keys():
#     print(key)

data = data['T']
#calculate variance before splitting into windows
variance = np.var(data)
#print("Variance: ", variance)

tot_win = np.sum(np.diff(np.r_[0, splice, T.size])//win_len)
X = np.zeros((tot_win, win_len))
start_arr = np.r_[0, splice]
end_arr = np.r_[splice, T.size]
start_x = 0
for start, end in zip(start_arr, end_arr):
    segment = T[start:end]
    n_win = segment.size//win_len
    i_win = np.arange(0, n_win*win_len, win_len)
    i_win = i_win[:, None] + np.arange(win_len)[None, :]
    X[start_x:start_x+n_win] = segment[i_win]
    start_x = start_x + n_win

k, P = args.num_clusters, args.centroid_len
metric, init = 'cosine', 'random'
n_runs, rng = 30, 13
centroids, labels, shifts, distances, _, _ = shift_invariant_k_means(
    X, k, P, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)


out_file = f'sikmeans_k-{k}_P-{P}_wlen-{win_len}.npz'
out_file_full = results_dir.joinpath(out_file)
with out_file_full.open('wb') as f:
    np.savez(f, centroids=centroids, labels=labels,
             shifts=shifts, distances=distances)

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')

#out_file = f'sikmeans_k-128_P-512_wlen-768.npz'
#out_file_full = results_dir.joinpath(out_file)

if args.visualize:
    with np.load(out_file_full) as data:
        centroids = data['centroids']
        labels = data['labels']
        shifts = data['shifts']
        distances = data['distances']

    unique_labels, cluster_size = np.unique(labels, return_counts=True)

    # Sort centroids in descending order of cluster size
    isort = np.argsort(-cluster_size)
    centroids = centroids[isort]
    unique_labels = unique_labels[isort]
    cluster_size = cluster_size[isort]
    # Determine the grid dimensions based on the number of centroids

    # determine number of centroids over some cluster size cutoff
    #args.visualize_cutoff = 5
    number = 0
    for i in cluster_size:
        if i >= args.visualize_cutoff:
            number += 1

    num_centroids = len(centroids)
    num_rows = int(np.ceil(np.sqrt(number)))
    num_cols = int(np.ceil(number / num_rows))

    # Create subplots with the determined grid dimensions
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axs array if necessary
    if num_centroids == 1:
        axs = np.array([axs])

    # Iterate over the centroids and plot each as a waveform in a separate subplot
    for i, centroid in enumerate(centroids):
        if cluster_size[i] >= 5:
            # Determine the subplot indices
            row_idx = i // num_cols
            col_idx = i % num_cols

            # Plot the waveform in the corresponding subplot
            axs[row_idx, col_idx].plot(centroid)
            # axs[row_idx, col_idx].set_title(f"Centroid {i + 1}")
            axs[row_idx, col_idx].set_title(cluster_size[i])

    # Remove empty subplots if the number of centroids is not a perfect square
    if num_centroids % num_cols != 0:
        for i in range(num_centroids, num_rows * num_cols):
            axs.flatten()[i].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot to images subdirectory
    img_dir = root.joinpath('img')#, args.experiment)
    img_dir.mkdir(exist_ok=True)
    img_dir_exp = img_dir.joinpath(args.experiment)
    img_dir_exp.mkdir(exist_ok=True)

    img_file = str(out_file).replace('.npz', '_img')