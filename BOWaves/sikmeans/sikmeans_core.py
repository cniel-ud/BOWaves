"""Shift-invariant k-means"""

import sys
import warnings
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sp

from sklearn.utils.extmath import stable_cumsum, squared_norm, row_norms
from sklearn.exceptions import ConvergenceWarning

from BOWaves.utilities import sikmeans_utils, wrappers



def run(experiment, root, centroid_len=512, win_len=768, num_clusters=128, visualize=True, visualize_cutoff=5):
    """

    Parameters
    ----------
    experiment: This is the name of the experiment and what you should call your subdirectory.
    root: Path to the root folder.
    centroid_len: Length of centroids. Default=512
    win_len: Non-overlapping window length. Default=768
    num_clusters: Number of clusters. Default=128
    visualize: Whether or not to visualize the results, output to /img. Default=True
    visualize_cutoff: Only centroids with this many occurences or above are visualized. Default=5

    Returns
    -------
    Returns nothing, but prints out values and saves visualizations and numpy data to img and results folders, respectively.
    """

    t_start = perf_counter()

    root = Path(root)
    data_dir = root.joinpath('data', experiment)
    data_dir.mkdir(exist_ok=True)

    results_dir = root.joinpath('results')
    results_dir.mkdir(exist_ok=True)

    results_dir = root.joinpath('results', experiment)
    results_dir.mkdir(exist_ok=True)

    fpath = list(data_dir.glob('*.npz'))[0]
    with np.load(fpath, allow_pickle=True) as data:
        T = data['T']
        splice = data['splice']

    data = np.load(fpath)

    data = data['T']
    # calculate variance before splitting into windows
    variance = np.var(data)

    tot_win = np.sum(np.diff(np.r_[0, splice, T.size]) // win_len)
    X = np.zeros((tot_win, win_len))
    start_arr = np.r_[0, splice]
    end_arr = np.r_[splice, T.size]
    start_x = 0
    for start, end in zip(start_arr, end_arr):
        segment = T[start:end]
        n_win = segment.size // win_len
        i_win = np.arange(0, n_win * win_len, win_len)
        i_win = i_win[:, None] + np.arange(win_len)[None, :]
        X[start_x:start_x + n_win] = segment[i_win]
        start_x = start_x + n_win

    k, P = num_clusters, centroid_len
    metric, init = 'cosine', 'random'
    n_runs, rng = 30, 13
    centroids, labels, shifts, distances, _, _ = shift_invariant_k_means(
        X, k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

    out_file = f'sikmeans_k-{k}_P-{P}_wlen-{win_len}.npz'
    out_file_full = results_dir.joinpath(out_file)
    with out_file_full.open('wb') as f:
        np.savez(f, centroids=centroids, labels=labels,
                 shifts=shifts, distances=distances)

    t_stop = perf_counter()
    print(f'Finished after {t_stop - t_start} seconds!')

    # out_file = f'sikmeans_k-128_P-512_wlen-768.npz'
    # out_file_full = results_dir.joinpath(out_file)

    if visualize:
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
        # args.visualize_cutoff = 5
        number = 0
        for i in cluster_size:
            if i >= visualize_cutoff:
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
        img_dir = root.joinpath('img')  # , args.experiment)
        img_dir.mkdir(exist_ok=True)
        img_dir_exp = img_dir.joinpath(experiment)
        img_dir_exp.mkdir(exist_ok=True)

        img_file = str(out_file).replace('.npz', '_img')

###############################################################################
# Initialization


def init_centroids(X, n_clusters, centroid_length, metric='euclidean',\
    init='k-means++', x_squared_norms=None, rng=None, **kwargs):
    """
    Compute initial centroids

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows are samples.
    n_clusters (int):
        Number of initial seed centroids
    centroid_length (int):
        Length of each centroid
    init ('k-means++', 'random', numpy.ndarray, or a function):
        Method for initialization. If it's a function, it should have this
        call signature:
        centroids, shifts = init(
            X, n_clusters, centroid_length, rng, **kwargs).
        rng must be a Generator instance.
    x_squared_norms (numpy.ndarray or None):
        Equivalent to np.matmul(X, X.T). If None, it would be computed if
        init='kmeans++'.
    rng (Int or Generator instance):
        The generator used to initialize the centroids. Use int to make the
        randomness deterministic.
    **kwargs:
        If init=='kmeans++', the following keyword argument can be used
            n_local_trials (int):
                The number of seeding trials for each centroid (except the first),
                of which the one reducing inertia the most is greedily chosen.
                Set to None to make the number of trials depend logarithmically
                on the number of seeds (2+log(k)); this is the default.


    Returns
    -------
    centroids (numpy.ndarray):
        The centroid seeds
    """

    rng = sikmeans_utils.check_rng(rng)

    n_samples, sample_length = X.shape

    if isinstance(init, str) and init == 'k-means++':
        if isinstance(metric, str) and metric == 'euclidean':
            if x_squared_norms is None:
                x_squared_norms = wrappers.si_row_norms(
                    X, centroid_length, squared=True)
            if len(kwargs) == 0 or 'n_local_trials' not in kwargs:
                n_local_trials = None
            else:
                n_local_trials = kwargs['n_local_trials']
            centroids = _kmeans_plus_plus(
                X, n_clusters, centroid_length, x_squared_norms, rng, n_local_trials)
        else:
            sys.exit("k-means++ is not implemented for non-euclidean metrics")
    elif isinstance(init, str) and init == 'random':
        centroids = _random_init(X, n_clusters, centroid_length,\
            rng)
    elif isinstance(init, str) and init == 'random-energy':
        centroids = _random_energy_init(
            X, n_clusters, centroid_length, rng)
    elif hasattr(init, '__array__'):
        # ensure that the centroids have the same dtype as X
        # this is a requirement of fused types of cython
        centroids = np.array(init, dtype=X.dtype)
    elif callable(init):
        centroids = init(X, n_clusters, centroid_length,
                         rng, **kwargs)
        centroids = np.asarray(centroids, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    return centroids


def _kmeans_plus_plus(X, n_clusters, centroid_length, x_squared_norms,\
    rng, n_local_trials=None):
    """
    Shift-invariant kmeans++

    This is a shift-invariant adapation to the implementation in scikit-learn.
    See http://bit.ly/sklearn_kmeans_pp

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows are samples.
    n_clusters (int):
        Number of initial seed centroids
    centroid_length (int):
        Lenght of each centroid
    x_squared_norms (numpy.ndarray):
        Equivalent to np.matmul(X, X.T)
    rng (Generator):
        The generator used to initialize the centroids.
    n_local_trials (int):
        The number of seeding trials for each centroid (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centroids (numpy.ndarray):
        The centroid seeds

    Notes
    -----
    Inertia, or potential, is the the sum of squared distances to the closest
    centroid, for all the samples.
    """

    n_samples, n_features = X.shape
    n_windows = n_features - centroid_length + 1

    centroids = np.empty((n_clusters, centroid_length), dtype=X.dtype)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first centroid randomly. Pick a sample, and get all the possible
    # windows (centroids). all_windows.shape=(1, n_windows, centroid_length).
    # XXX: Use a simpler function to get all the windows. They don't need to be
    # randomly selected.
    sample_id = rng.integers(n_samples)
    if sp.issparse(X):
        all_windows = sikmeans_utils.pick_random_windows(X[sample_id].toarray(),
                                                         n_windows, centroid_length,
                                                         rng)
    else:
        all_windows = sikmeans_utils.pick_random_windows(X[sample_id], n_windows,
                                                         centroid_length, rng)
    all_windows = all_windows.squeeze(axis=0)

    # Initialize list of closest distances.
    # closest_dist_sq.shape=(n_windows,n_windows,centroid_length). The first
    # dimension is the number of shifts of each window, which is also equal to
    # the number of windows.
    closest_dist_sq = wrappers.si_euclidean_distances(
        all_windows, X, x_squared_norms, squared=True)

    # Potential: the sum of squared distances to closest centroid
    # Compute potential for each shift and each window
    current_pot = closest_dist_sq.sum(axis=2)

    # Find best window and its best shift
    best_shift_id, best_window_id = np.unravel_index(
        np.argmin(current_pot), current_pot.shape)

    # Update distances and potential to use best window and best shift
    closest_dist_sq = closest_dist_sq[best_shift_id, best_window_id]
    current_pot = current_pot[best_shift_id, best_window_id]

    # Update centroids and shifts
    centroids[0] = all_windows[best_window_id]

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose centroid candidates by sampling with probability proportional
        # to the squared distance to the closest existing centroid
        rand_vals = rng.random(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Pick all windows from each candidate sample.
        # all_windows.shape=(n_local_trials,n_windows,centroid_length)
        if sp.issparse(X):
            all_windows = sikmeans_utils.pick_random_windows(
                X[candidate_ids].toarray(), n_windows, centroid_length,
                rng)
        else:
            all_windows = sikmeans_utils.pick_random_windows(
                X[candidate_ids], n_windows, centroid_length, rng)

        # Compute distances to centroid candidates for all windows
        distance_to_candidates = np.empty(
            (n_local_trials, n_windows, n_windows, n_samples))
        for trial in np.arange(n_local_trials):
            distance_to_candidates[trial] = \
                wrappers.si_euclidean_distances(
                    all_windows[trial], X, X_norm_squared=x_squared_norms,
                    squared=True)

        # Update the distances so that distance_to_candidates[i][j][k][l] has the distances that the problem would reach if the k-th window of the i-th candidate sample is selected.
        np.minimum(distance_to_candidates, closest_dist_sq,
                   out=distance_to_candidates)

        # Compute potential.
        # candidates_pot.shape=(n_local_trials, n_windows, n_windows)
        candidates_pot = distance_to_candidates.sum(axis=3)

        # Find best candidate, window and shift
        min_pot_id = np.argmin(candidates_pot)
        best_candidate_id, best_shift_id, best_window_id = np.unravel_index(
            min_pot_id, candidates_pot.shape)

        # Choose best potential and distances
        current_pot = candidates_pot[best_candidate_id,
                                     best_shift_id,
                                     best_window_id]
        closest_dist_sq = distance_to_candidates[best_candidate_id,
                                                 best_shift_id,
                                                 best_window_id]

        # Permanently add best centroid candidate
        centroids[c] = all_windows[best_candidate_id, best_window_id]

    return centroids

def _random_init(X, n_clusters, centroid_length, rng):
    n_samples = X.shape[0]
    seeds = rng.permutation(n_samples)[:n_clusters]
    centroids = X[seeds]
    centroids = sikmeans_utils.pick_random_windows(centroids, 1, centroid_length,
                                                   rng).squeeze()

    return centroids


def _random_energy_init(X, n_clusters, centroid_length, rng):
    n_samples = X.shape[0]
    seeds = rng.permutation(n_samples)[:n_clusters]
    windows = X[seeds]
    # Pick all windows
    # (n_rows, n_offsets, window_length)
    windows = sikmeans_utils.pick_windows(windows, centroid_length, offset='all')
    shifts = np.empty(n_clusters, dtype=np.int64)
    for i_centroid in range(n_clusters):
        energy = row_norms(windows[i_centroid], squared=False)
        shifts[i_centroid] = np.argmax(energy)

    ind1 = np.arange(n_clusters)[:, None]
    ind2 = shifts[:, None]
    ind3 = np.arange(centroid_length)[None, :]
    centroids = windows[ind1, ind2, ind3]

    return centroids


###############################################################################
# Main algorithm

def shift_invariant_k_means(X, n_clusters, centroid_length, metric='euclidean',\
    init='k-means++', n_init=10, max_iter=300, tol=1e-4, rng=None, verbose=False):
    """
    Shift-invariant k-means algorithm

    Parameters
    ----------
    X (numpy.ndarray):
        Data matrix with samples in its rows.
    n_clusters (int):
        Number of clusters to form, as well as the number of centroids to find.
    centroid_length (int):
        The length of each centroid.
    metric ('euclidean' or 'cosine'):
        Metric used to compute the distance between samples and cluster centroids. Default: 'euclidean'.
    init ('k-means++', 'random', numpy.ndarray, or a function):
        Method for initialization. If it's a function, it should have this
        call signature:
        centroids, shifts = init(
             X, n_clusters, centroid_length, rng, **kwargs).
        rng must be a Generator instance.
    n_init (int):
        The number of times the algorithm is run with different centroid seeds.
        The final results would be from the iteration where the inertia is the
        lowest.
    max_iter (init):
        Maximum number of iterations the algorithm will be run.
    tol (float):
        Upper bound that the squared euclidean norm of the change in the
        centroids must achieve to declare convergence.
    rng (int, Generator instance or None):
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.
    verbose (bool):
        If True, print details about each iteration.

    Returns
    -------
    centroids (numpy.ndarray):
        A matrix with the learned centroids in its rows.
    labels (numpy.ndarray):
        labels[i] is the index of the centroid (row of `centroids`) closest
        to the sample X[i].
    shifts (numpy.ndarray):
        shift[i] is the shift that minimizes the distance to the closest
        centroid to the sample X[i].
    distances (numpy.ndarray):
        distances[i] is the distance from X[i,shift[i]:shift[i]+centroid_length]
        to its closest centroid.
    inertia (float):
        The sum of squared euclidean distances to the closest centroid of all the
        training samples.
    best_n_iter (int):
        Number of iterations needed to achieve convergence, according to `tol`.
    """

    rng = sikmeans_utils.check_rng(rng)

    best_labels, best_shifts, best_centroids = None, None, None
    best_distances, best_inertia, best_n_iter = None, None, None

    # subtract of mean of x for more accurate distance computations
    # NOTE: Can't do that because each centroid is the average of windows from X
    # that were chosen at different starting times.

    # Precompute squared norms of rows of X for efficient computation of
    # euclidean distances between centroids and samples. Do this for each set of
    # windows (shifts) of X. x_squared_norms.shape=(n_shifts, n_samples).
    x_squared_norms = None
    if isinstance(metric, str) and metric == 'euclidean':
        x_squared_norms = wrappers.si_row_norms(X, centroid_length,
                                                squared=True)


    ss = rng.bit_generator._seed_seq
    child_seeds = ss.spawn(n_init)
    streams = [np.random.default_rng(s) for s in child_seeds]

    for seed in streams:
        # run a shift-invariant k-means once
        centroids, labels, shifts, distances, inertia, n_iter_ = si_kmeans_single(
            X, n_clusters, centroid_length, metric=metric, init=init, max_iter=max_iter, tol=tol, x_squared_norms=x_squared_norms, rng=seed, verbose=verbose)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_distances = distances
            best_inertia = inertia
            best_n_iter = n_iter_

    distinct_clusters = len(set(best_labels))

    if distinct_clusters < n_clusters:
        warnings.warn(
            "Number of distinct clusters ({}) found smaller than "
            "n_clusters ({}). Possibly due to duplicate points "
            "in X.".format(distinct_clusters, n_clusters), ConvergenceWarning,
            stacklevel=2
        )

    return best_centroids, best_labels, best_shifts, best_distances, best_inertia, best_n_iter


def si_kmeans_single(X, n_clusters, centroid_length, metric='euclidean',\
    init='k-means++', max_iter=300, tol=1e-3, x_squared_norms=None, rng=None, verbose=False):
    """
    Single run of shift-invariant k-means
    """

    rng = sikmeans_utils.check_rng(rng)

    if isinstance(metric, str) and metric == 'euclidean':
        if x_squared_norms is None:
            x_squared_norms = wrappers.si_row_norms(
                X, centroid_length, squared=True)

    best_labels, best_shifts, best_centroids = None, None, None
    best_distances, best_inertia = None, None

    # Init
    centroids = init_centroids(
        X, n_clusters, centroid_length, metric, init, x_squared_norms, rng)

    #The below is Dr. B's additions from the Jupyter notebook.
    #I've added the update step function to the utils file.
    #Adding here to test before PR
    labels, shifts, distances = _assignment_step(
        X, centroids, metric, x_squared_norms)
    centroids = _init_centroids_update_step(
        X, centroid_length, n_clusters, labels, shifts) # NEW

    if verbose:
        print('Initialization completed.')

    for iteration in range(max_iter):
        centroids_old = centroids.copy()
        labels, shifts, distances = _assignment_step(X, centroids, metric, x_squared_norms)
        centroids = _centroids_update_step(
            X, centroid_length, n_clusters, labels, shifts)

        inertia = distances.mean()

        if verbose:
            print("Iteration %2d, inertia %.3f" % (iteration, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_centroids = centroids.copy()
            best_distances = distances
            best_inertia = inertia

        centroid_change = squared_norm(centroids_old - centroids)/n_clusters/centroid_length
        #print(centroid_change, tol)
        if centroid_change <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "centroid changes %e within tolerance %e"
                      % (iteration, centroid_change, tol))
            break

    if centroid_change > 0:
        # rerun asingment step in case of non-convergence so that predicted
        # labels match cluster centers
        best_labels, best_shifts, distances = _assignment_step(X, best_centroids, metric, x_squared_norms)
        best_inertia = distances.mean()

    return best_centroids, best_labels, best_shifts, best_distances, best_inertia, iteration+1


def _assignment_step(X, centroids, metric, x_squared_norms):
    """
    Find the index of the shifted centroid that is closest to each sample

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows of X are samples.
    centroids (numpy.ndarray):
        Centroids of the clusters.
    x_squared_norms (numpy.ndarray):
        Squared Euclidean norm of rows of X. This is used to speed up the
        computation of the Euclidean distances between samples and centroids.

    Returns
    -------
    labels (numpy.ndarray):
        centroids[labels[i]] is the centroid closest to sample X[i]
    shifts (numpy.ndarray):
        X[i, shifts[i]:shifts[i]+centroid_length] is the window in X[i]  closest to centroids[labels[i]].
    distances (numpy.ndarray):
        distances[i] is the distance of X[i, shifts[i]:shifts[i]+ centroid_length] to the closest centroid.
    """

    #for testing
    #metric = 'euclidean'

    labels, shifts, distances =\
        wrappers.si_pairwise_distances_argmin_min(
            X, centroids, metric, x_squared_norms)

    # Samples whose distance to the silent waveform (i.e, their own norm) is
    # smaller than the smallest distance to one cluster are left unassigned.
#    xsn = x_squared_norms[shifts, np.arange(X.shape[0])]
#    discard_ind=np.where(xsn < distances)[0]
#    labels[discard_ind] = -1

    return labels, shifts, distances


def _centroids_update_step(X, centroid_length, n_clusters, labels, shifts):
    """
    Update the cluster centroids
    """

    centroids = np.zeros((n_clusters, centroid_length))

    for sample_id, sample in enumerate(X):
        cluster_id = labels[sample_id]
        shift = shifts[sample_id]
        centroids[cluster_id] += sample[shift:shift+centroid_length]

    # NOTE: Some clusters might be empty
    cluster_id, cluster_size = np.unique(labels, return_counts=True)
    centroids[cluster_id, :] /= cluster_size[:, np.newaxis]

    return centroids


def _init_centroids_update_step(X, centroid_length, n_clusters, labels, shifts):
    """
    Update the cluster centroids
    """

    cluster_ids, _ = np.unique(labels, return_counts=True)
    centroids = np.zeros((n_clusters, centroid_length))
    n_samples, sample_length = X.shape
    # adjust the shifts such that after adjustment the median shift is
    max_shift = sample_length - centroid_length
    opt_shift = max_shift/2
    adjusts = np.zeros((n_clusters))
    for k in cluster_ids:
        shifts_k = shifts[labels==k]
        adjusts[k] = opt_shift-np.median(shifts_k)

    cluster_sizes = np.zeros((n_clusters,1))
    for sample_id, sample in enumerate(X):
        cluster_id = labels[sample_id]
        temp = shifts[sample_id]+adjusts[cluster_id]
        if temp >= 0 and temp <= max_shift:
            shift = np.floor(temp).astype(int)
            centroids[cluster_id] += sample[shift:shift+centroid_length]
            cluster_sizes[cluster_id] += 1

    # NOTE: Some clusters might be empty drop them
    #centroids/= cluster_sizes
    for k in np.where(cluster_sizes==0)[0]:
        centroids[k,:] = 0
    for k in np.nonzero(cluster_sizes)[0]:
        centroids[k,:]/= cluster_sizes[k]

    return centroids