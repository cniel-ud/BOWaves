"""
Bilal wants one subject's brain waves and the associated counts.

This file will get cue dataset's subj 1 brain waves and the count vector.
The cue dataset's native sampling rate is 500 Hz, and I will resample to the sampling rate of the mice data - 256 Hz.

This will also serve as an API guide for Bilal when he looks further into the code and writes his paper's method section.
"""
import pyrootutils

pyrootutils.set_root(path='/work/cniel/ajmeek/BOWaves/BOWaves', pythonpath=True)

from BOWaves.utilities.dataloaders import load_codebooks, load_raw_set, load_raw_set_single_subj
from BOWaves.bowav.bowav_feature_extractor import bag_of_waves
import BOWaves.utilities.dataloaders as dataloaders
from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means, _assignment_step

import scipy
from numpy.random import default_rng
import numpy as np
from pathlib import Path

# # resample cue subj 01 to 256 hz
# matdict = scipy.io.loadmat('../data/codebooks/frolich/subj-01.mat')
#
# # signals stored in 'data' field
# # original length is 1979155. So (500/256) * 1979155
# matdict['data'] = scipy.signal.resample(matdict['data'], int((500.0/256) * 1979155), axis=1)
#
# # save matdict to new file
# scipy.io.savemat('../data/codebooks/frolich/subj-01_resampled_to_mice.mat', matdict)


# intermission - switch to HPC here to run codebooks. Then come back to desktop to run BOWaves
# code run in between these lines was done on Caviness / desktop
# ----------------------------------------------------------------------------------------------
file = '../data/cue/cue_signals_resampled_to_emotion/subj-01_resampled_to_mice.mat'

# these hyperparams are based on sampling rate. want 1 sec centroid, 1.5 sec window
window_len = 750
metric, init = 'cosine', 'random'
num_clusters = 16
centroid_len = 500
n_runs = 3
n_jobs = 1
rng = default_rng()

# hold classified ICs. Technically here we only want the brain ones, but this is boilerplate at the moment
neural = {'name': 'neural', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
blink = {'name': 'blink', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
muscle = {'name': 'muscle', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
mixed = {'name': 'mixed', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
lateyes = {'name': 'lateyes', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
heart = {'name': 'heart', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}

all_classes = [neural, blink, muscle, mixed, lateyes, heart]

ICs, labels = dataloaders.load_and_visualize_mat_file_frolich(file, visualize=False)

# error check
if len(ICs) != len(labels):
    raise ValueError('ICs and labels are not the same length.')

for i, label in enumerate(labels):
    if label == 'neural':
        neural['ICs'].append(ICs[i])
    elif label == 'blink':
        blink['ICs'].append(ICs[i])
    elif label == 'muscle':
        muscle['ICs'].append(ICs[i])
    elif label == 'mixed':
        mixed['ICs'].append(ICs[i])
    elif label == 'lateyes':
        lateyes['ICs'].append(ICs[i])
    elif label == 'heart':
        heart['ICs'].append(ICs[i])
    else:
        raise ValueError('Unknown class label: ' + label)

# train codebook for neural data
neural['centroids'], neural['labels'], neural['shifts'], neural['distances'], neural['inertia'] = \
    shift_invariant_k_means(neural['ICs'], num_clusters, window_len, metric, init, n_runs, n_jobs, rng)

# save codebook
np.savez(f'../data/codebooks/frolich/sikmeans_P-{window_len}_k-{num_clusters}_class-neural_subj-1.npz',
         centroids=neural['centroids'], labels=neural['labels'], shifts=neural['shifts'], distances=neural['distances'],
         inertia=neural['inertia'])


# ----------------------------------------------------------------------------------------------

# # now load codebook and calculate the BOWav count vector from it
# class args:
#     root = '../data/codebooks/frolich'
#     num_clusters = 16
#     window_len = 750
#     minutes_per_ic = 1.5
#     ics_per_subject = 100
#     num_classes = 6
#     centroid_len = 500
#     n_jobs = 1
#     rng = default_rng()
# codebooks = load_codebooks(args)
# rng = default_rng()
#
# raw_ics, y, expert_label_mask, \
#     subj_ind, noisy_labels, labels = load_raw_set_single_subj(args, rng, Path('../data/codebooks/frolich'), fnames=['subj-01_resampled_to_mice.mat'])
#
# X = bag_of_waves(raw_ics, codebooks)
#
# # save BOWav count vector to a file
# np.savez('../data/codebooks/frolich/bowav_count_vector_subj-01.npz', X=X, y=y, expert_label_mask=expert_label_mask,
#          subj_ind=subj_ind, noisy_labels=noisy_labels, labels=labels)