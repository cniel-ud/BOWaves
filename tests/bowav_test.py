"""
Test for BoWav feature extractor.

Get ICs from sample data, then test what BoWav finds. If it makes sense, pass the test.

"""

from pathlib import Path
import numpy as np
import re
from scipy.io import loadmat
from tqdm import tqdm
from numpy.random import default_rng
import BOWaves
from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means
from BOWaves.bowav.bowav_feature_extractor import bag_of_waves

#change to your root directory to run the test. Hardcoded here since this is a simple sanity check, not included in the package
root = '/home/austin/PycharmProjects/BOWaves'

root = Path(root)
data_dir = root.joinpath('data', 'morlet')
data_dir.mkdir(exist_ok=True)

results_dir = root.joinpath('results')
results_dir.mkdir(exist_ok=True)

results_dir = root.joinpath('results', 'morlet')
results_dir.mkdir(exist_ok=True)

fpath = list(data_dir.glob('*.npz'))[0]
with np.load(fpath, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

#print(T.shape)

ds003004_data_path = Path('/home/austin/PycharmProjects/BOWaves/data/bowav_test_ics')
rng = default_rng(13)

#Carlos' dictionary creation code from ICWaves repo, and the dataloader necessary for it
#again, to be added as part of a broader pipeline later

def load_raw_train_set_per_class(srate, window_len, minutes_per_ic):

    #default arg value for things like class_label in Carlos' code was '1'. So will replace things accordingly with defaults.

    data_dir = ds003004_data_path
    file_list = list(data_dir.glob(f'train_subj-*.mat')) #for now this is training data. what I got from Isabel

    ic_ind_per_subj = []
    subj_with_no_ic = []
    for i_subj, file in enumerate(file_list):
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=[
                              'expert_labels', 'noisy_labels'])
            expert_labels = matdict['expert_labels']
            noisy_labels = matdict['noisy_labels']

        if 1 in EXPERT_ANNOTATED_CLASSES:
            ic_ind = (expert_labels == 1).nonzero()[0]
        else:
            winner_class = np.argmax(noisy_labels, axis=1)
            winner_class = winner_class + 1  # python to matlab indexing base
            ic_ind = (winner_class == 1).nonzero()[0]

        if ic_ind.size > 0: # subject has IC class
            ic_ind_per_subj.append(ic_ind)
        else:
            subj_with_no_ic.append(i_subj)

    file_list = [file for i, file in \
        enumerate(file_list) if i not in subj_with_no_ic]

    n_subj = len(ic_ind_per_subj)
    n_ics_per_subj = np.array(list(map(lambda x: x.size, ic_ind_per_subj)))
    subj_with_ic_excess = (n_ics_per_subj > 2).nonzero()[0]
    n_ics_per_subj[subj_with_ic_excess] = 2
    n_ics = np.sum(n_ics_per_subj)

    for i_subj in subj_with_ic_excess:
        ic_ind_per_subj[i_subj] = rng.choice(
            ic_ind_per_subj[i_subj], size=2, replace=False)

    icaact_list = [None] * n_subj
    for i_subj, file in tqdm(enumerate(file_list)):
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names='icaact')
            icaact = matdict['icaact']

        icaact_list[i_subj] = icaact[ic_ind_per_subj[i_subj]]

    # ICs from different subjects have different lenths, so we don't
    # concatenate into a single array
    ic_length_per_subj = np.array(list(map(lambda x: x.shape[1], icaact_list)))
    max_n_win_per_subj = ic_length_per_subj // window_len
    max_minutes_per_ic_per_subj = (max_n_win_per_subj *
                                   window_len) / srate / 60
    min_max_minutes_per_ic = 50#np.min(max_minutes_per_ic_per_subj)

    take_all = True
    if (
        minutes_per_ic is not None and
        minutes_per_ic < min_max_minutes_per_ic
    ):
        minutes_per_window = (window_len/srate/60)
        n_win_per_ic = np.ceil(minutes_per_ic /
                            minutes_per_window).astype(int)
        take_all = False
    else:
        n_win_per_ic = ic_length_per_subj // window_len

    tot_win = (n_win_per_ic * n_ics_per_subj).sum().astype(int)
    tot_hrs = tot_win * window_len / srate / 3600

    print(f"Training ICs for '{CLASS_LABELS[1-1]}': {n_ics}")
    print(f"Number of training hours: {tot_hrs:.2f}")

    X = np.zeros((tot_win, window_len))#, dtype=icaact_list[0].dtype)
    win_start = 0
    for i_subj, ics in tqdm(enumerate(icaact_list)):
        n_win = n_win_per_ic[i_subj] if take_all else n_win_per_ic
        for ic in ics:
            time_idx = np.arange(0, ic.size-window_len+1, window_len)
            time_idx = rng.choice(time_idx, size=n_win, replace=False)
            time_idx = time_idx[:, None] + np.arange(window_len)[None, :]
            X[win_start:win_start+n_win] = ic[time_idx]
            win_start += n_win

    return X


def load_codebooks():

    dict_dir = Path('/home/austin/PycharmProjects/BOWaves/results/bowav_test_dictionaries')

    n_codebooks = 7
    num_clusters = 16
    centroid_len = 256
    codebooks = np.zeros((n_codebooks, num_clusters,
                        centroid_len), dtype=np.float32)

    for i_class in range(n_codebooks):
        fname = (
            f'sikmeans_P-{centroid_len}_k-{num_clusters}'
            f'_class-{i_class+1}_minutesPerIC-{50}'
            f'_icsPerSubj-{2}.npz'
        )
        fpath = dict_dir.joinpath(fname)
        with np.load(fpath) as data:
            codebooks[i_class] = data['centroids']

    return codebooks

def dictionary():
    srate, win_len = 256, 384
    rng = default_rng(13)
    num_clusters = 16
    centroid_len = 256
    n_runs = 3
    n_jobs = 1
    centroid_len = 256
    class_label = 1
    minutes_per_ic = 50
    ics_per_subject = 2

    X = load_raw_train_set_per_class(srate, win_len, minutes_per_ic)

    metric, init = 'cosine', 'random'
    centroids, labels, shifts, distances, inertia, _ = shift_invariant_k_means(
        X, num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)

    dict_dir = Path('/home/austin/PycharmProjects/results/bowav_test_dictionaries')
    out_file = dict_dir.joinpath(
        f'sikmeans_P-{centroid_len}_k-{num_clusters}'
        f'_class-{class_label}_minutesPerIC-{minutes_per_ic}'
        f'_icsPerSubj-{ics_per_subject}.npz'
    )
    with out_file.open('wb') as f:
        np.savez(out_file, centroids=centroids, labels=labels,
                shifts=shifts, distances=distances, inertia=inertia)

#Carlos' data loading code from his matlab processing outputs

EXPERT_ANNOTATED_CLASSES = [1, 2, 3]  # brain, muscle, eye (Matlab indexing)

CLASS_LABELS = ['Brain', 'Muscle', 'Eye', 'Heart',
                'Line Noise', 'Channel Noise', 'Other']

def load_raw_set(data_dir, win_len, srate, min_per_ic, rng, train=True):

    file_prefix = 'train' if train else 'test'
    #data_dir = Path(args.root, 'data/ds003004/icact_iclabel')
    file_list = list(data_dir.glob(f'{file_prefix}_subj-*.mat'))

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names='expert_labels')
            expert_labels = matdict['expert_labels']
            n_ics_per_subj.append(expert_labels.shape[0])

    n_ics = np.sum(n_ics_per_subj).astype(int)
    minutes_per_window = (win_len/srate/60)
    n_win_per_ic = np.ceil(min_per_ic / minutes_per_window).astype(int)

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_win_per_ic, win_len), dtype=np.float32)
    y = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask = np.full(n_ics, False)
    subj_ind_ar = np.zeros(n_ics, dtype=int)
    noisy_labels_ar = []
    p = re.compile(f'.+{file_prefix}_subj-(?P<subjID>\d{{2}}).mat')
    for file in tqdm(file_list):
        with file.open('rb') as f:
            matdict = loadmat(f)
            expert_labels = matdict['expert_labels']
            icaact = matdict['icaact']
            noisy_labels = matdict['noisy_labels']

        noisy_labels_ar.append(noisy_labels)
        m = p.search(str(file))
        subjID = int(m.group('subjID'))

        ics_with_expert_label = (expert_labels > 0).nonzero()[0]
        for ic_ind, ic in enumerate(icaact):
            time_idx = np.arange(0, ic.size-win_len+1, win_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(win_len)[None, :]
            X[cum_ic_ind] = ic[time_idx]

            subj_ind_ar[cum_ic_ind] = subjID

            if ic_ind in ics_with_expert_label:
                # -1: Let class labels start at 0 in python
                y[cum_ic_ind] = expert_labels[ic_ind] - 1
                expert_label_mask[cum_ic_ind] = True
            else:
                noisy_label = np.argmax(noisy_labels[ic_ind])
                y[cum_ic_ind] = noisy_label
            cum_ic_ind += 1

    #noisy_labels_ar = np.vstack(noisy_labels_ar)

    return X, y, expert_label_mask, subj_ind_ar, noisy_labels_ar

ds003004_data_path = Path('/home/austin/PycharmProjects/BOWaves/data/bowav_test_ics')
rng = default_rng(13)
X, y, expert_label_mask, subj_ind_ar, noisy_labels_ar = load_raw_set(ds003004_data_path, win_len=384, srate=256, min_per_ic=50, rng=rng)

dictionary()
codebooks = load_codebooks()

print(bag_of_waves(X, codebooks))

#note, X above is what we want for the bowav test. Of shape (n_ics, n_win_per_ic, win_len)

