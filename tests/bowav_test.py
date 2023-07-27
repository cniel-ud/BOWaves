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

#note, X above is what we want for the bowav test. Of shape (n_ics, n_win_per_ic, win_len)

