"""
This file is for helper functions used for loading in EEG ICs.

Current pipeline:
1. Run EEG dataset through EEGlab to get ICs - this requires Matlab.
    If you're at UD, this can be done on Caviness if your computer doesn't have Matlab installed.

2. Run the sikmeans core run utility - pass in the name of the data folder for the "experiments" parameter.
    This will most likely be named in accordance with BIDS convention - i.e., 'ds003004'

3. Take the results from the above function for use in bag of waves.

4. Features from BOWaves can then be used for classification - clustering for dictionary learning.

Where the functions housed in the current file come into play in the above pipeline:
2. Load ICs, load from matlab file outputs using scipy, use sklearn for train test split
3. Dictionary and codebook creation files for BOWaves. Associated dataloaders
"""

from scipy.io import loadmat
from pathlib import Path
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt
import os

def load_ics_from_matlab(root: Path, dataset_name: str):
    """
    This function loads ICs from the data folder and returns a numpy array with them.

    # TODO - before writing any more of this, run the matlab code and see how its outputs are formatted

    Parameters
    ----------
    root
    dataset_name

    Returns
    -------
    numpy array of ICs. Of shape (channel, time, # of ICs)
    """

    data_dir = root.joinpath('data', dataset_name)


def load_and_visualize_mat_file_frolich(file_path, up_to=None, visualize=False, cue=False):
    """
    This takes in the preprocessed data from Frolich et. al
    W is the demixing matrix from ICA, X is the array of ICs
    Classes and labels are in nested arrays, which explains the weird and complicated indexing below.
    Check the data array in the debugger if you want more details.
    Parameters
    ----------
    file_path: path to the .mat file containing the data

    visualize: Boolean, whether or not to use matplotlib to visualize the data and save it to /img subdirectory

    up_to: int, number of ICs to load. If None, load all ICs. For testing, can load smaller subset.

    Returns
    -------
    Y - a matrix of ICs. Shape is (channels, samples)
        For the Frolich data, there are around 2 mil samples, at 500 hz sampling rate
    """
    # Create 'img' subdirectory if it doesn't exist
    #img_dir = os.path.join(os.path.dirname(file_path), 'img')
    #os.makedirs(img_dir, exist_ok=True)

    # Load .mat file
    data = loadmat(file_path)

    # Display metadata
    # print("Metadata:")
    # for key, value in data.items():
    #     if not key.startswith("__"):
    #         print(f"{key}: {type(value)}")
    #         # if isinstance(value, np.ndarray):
    #         #     print(value)

    # if not cue:
    # Visualize EEG time series data
    X = data['X'] #raw
    W = data['W'] #demixing matrix

    Y = W @ X #combine here to get the ICs
    # elif cue:
    #     X = data['data']
    #     sphere = data['icasphere']
    #     weights = data['icaweights']
    #
    #     Y = weights @ sphere @ X

    # this is the Cue dataset from Frolich, not the Emotion one. 500 Hz
    # train classifier on emotion, test on Cue. need to change sampling rate in between

    # need different number of minutes per IC / window. Carlos' default params were 15, which is 27 mil
    # time points. We've only got 2 mil. Change that param based on what Frolich uses and also keep in mind
    # what ICLabel uses.

    # if visualize:
    #     num_channels, num_samples = Y.shape
    #     time = np.arange(num_samples)  # Assuming time starts from 0 and is evenly spaced
    #
    #     fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))
    #     fig.suptitle('Independent Components')
    #
    #     for channel in range(num_channels):
    #         axes[channel].plot(time // 800 , Y[channel, time // 800]) #get 2500 samples from 2 mil
    #         #axes[channel].set_ylabel(f'Channel {channel + 1}')
    #         #if not (key == "X" and channel == 63):
    #         #print(channel)
    #         axes[channel].set_ylabel(f'Channel {channel} \n Label {data["classes"][0][data["labels"][channel] - 1][0][0]}')
    #
    #     axes[-1].set_xlabel('Time')
    #     plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
    #
    #     # Save the plot to the 'img' subdirectory
    #     plot_filename = os.path.join(img_dir, f"Y_plot.png")
    #     plt.savefig(plot_filename)
    #     plt.close()
    #
    #     plt.show()

    labels_in_order = []
    for count, i in enumerate(data['labels']):
        # print(f"IC #{count} is label {data['classes'][0][i-1][0][0]}")
        # if not cue:
        labels_in_order.append(data['classes'][0][i-1][0][0])
        # elif cue:
        #     if i == 1:
        #         labels_in_order.append('blink')
        #     elif i == 2:
        #         labels_in_order.append('neural')
        #     elif i == 3:
        #         labels_in_order.append('heart')
        #     elif i == 4:
        #         labels_in_order.append('lateyes')
        #     elif i == 5:
        #         labels_in_order.append('muscle')
        #     elif i == 6:
        #         labels_in_order.append('mixed')
        #     else:
        #         # non-cue data (emotion) has different set of labels with len 7 != 6.
        #         pass

    return Y, labels_in_order

# Replace 'your_file.mat' with the actual file path
#Y, labels = load_and_visualize_mat_file_frolich('../../data/frolich/frolich_extract_01.mat')#, visualize=True)

#print()

def load_codebooks(args):#dict_dir, num_clusters, centroid_len, minutes_per_ic, ics_per_subject):
    """

    This loads the codebooks. Assume that the codebooks are all housed in the results/dictionaries
    folder and that we can then find them based off of the args passed in.

    Parameters
    ----------
    dict_dir: For testing cue dataset, expected to be root / results / emotion_clf_dictionaries
        This should be a Path object, not a string.
    num_clusters
    centroid_len
    minutes_per_ic
    ics_per_subject

    Returns
    -------

    """

    dict_dir = Path('../data/codebooks/emotion')

    n_codebooks = 7
    codebooks = np.zeros((n_codebooks, args.num_clusters,
                        args.centroid_len), dtype=np.float32)

    for i_class in range(n_codebooks):
        fname = (
            f'sikmeans_P-256_k-{args.num_clusters}' 
            f'_class-{i_class+1}_minutesPerIC-{args.minutes_per_ic}'
            f'_icsPerSubj-{args.ics_per_subject}.npz'
        )
        fpath = dict_dir.joinpath(fname)
        with np.load(fpath) as data:
            codebooks[i_class] = data['centroids']

    return codebooks


def load_codebooks_resampled(args):#dict_dir, num_clusters, centroid_len, minutes_per_ic, ics_per_subject):
    """

    This loads the codebooks. Assume that the codebooks are all housed in the results/dictionaries
    folder and that we can then find them based off of the args passed in.

    When we use this in the package, the root / results will be different. Use highest level results
    directory for what we use to test the package.
    Let me just put in the directory now.


    If package keeps being developed, then eventually unit tests and whatnot will all use those results.


    Parameters
    ----------
    dict_dir: For testing cue dataset, expected to be root / results / emotion_clf_dictionaries
        This should be a Path object, not a string.
    num_clusters
    centroid_len
    minutes_per_ic
    ics_per_subject

    Returns
    -------

    """
    #
    # n_codebooks = 7
    # codebooks = np.zeros((n_codebooks, num_clusters, centroid_len), dtype=np.float32)
    #
    # for i_class in range(n_codebooks):
    #     fname = (
    #         f'sikmeans_P-{centroid_len}_k-{num_clusters}'
    #         f'_class-{i_class+1}_minutesPerIC-{minutes_per_ic}'
    #         f'_icsPerSubj-{ics_per_subject}.npz'
    #     )
    #
    #     fpath = dict_dir.joinpath(fname)
    #     with np.load(fpath) as data:
    #         codebooks[i_class] = data['centroids']
    #
    # return codebooks


    # TODO - hardcoded stuff for testing clf. change later for generalization

    dict_dir = Path(args.root, '/data/codebooks/emotion_resampled_to_cue')

    dict_dir = Path('../data/codebooks/emotion_resampled_to_cue')

    n_codebooks = 7
    codebooks = np.zeros((n_codebooks, args.num_clusters,
                        args.centroid_len), dtype=np.float32)

    for i_class in range(n_codebooks):
        fname = (
            f'sikmeans_P-256_k-{args.num_clusters}' #actually P is 500 but didn't change filename when resampled
            f'_class-{i_class+1}_minutesPerIC-{args.minutes_per_ic}'
            f'_icsPerSubj-{args.ics_per_subject}_resampled.npz'
        )
        fpath = dict_dir.joinpath(fname)
        with np.load(fpath) as data:
            codebooks[i_class] = data['centroids']

    return codebooks


# Note, should overload this func to take a subj_id or not
def load_raw_set(args, rng, subj_ids):
    data_dir = Path(args.root, '/data/cue/')
    # the above line does not work on Caviness. I don't know why.
    # therefore,
    data_dir = Path('../data/cue')


    fnames = [f"subj-{i}.mat" for i in subj_ids] #modify to test on subset for smaller time

    #temp fix, change to all subjects later
    #fnames = fnames[0] #test a single subject for now
    file_list = [data_dir.joinpath(f) for f in fnames]
    # fnames = f'subj-{subj_id}.mat'
    # file_list = [data_dir.joinpath(fnames)]

    print("data_dir: ", data_dir)
    print("file_list:\n\t", file_list)

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=['labels', 'srate'])
            labels = matdict['labels']
            srate = matdict['srate']  # assumes all subjects have the same sampling rate
            srate = srate.item(0)  # `srate.shape=(1,1)`. This extracts the number.
            n_ics_per_subj.append(labels.shape[0])

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = (args.window_len / srate / 60)
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_win_per_ic, args.window_len), dtype=np.float32)
    y = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask_ar = np.full(n_ics, False)
    subj_ind = np.zeros(n_ics, dtype=int)
    # 7 ICLabel classes
    noisy_labels_ar = np.zeros((n_ics, 7), dtype=np.float32)
    for file, subjID in tqdm(zip(file_list, args.subj_ids)):
        with file.open('rb') as f:
            matdict = loadmat(f)
            data = matdict['data']
            icaweights = matdict['icaweights']
            icasphere = matdict['icasphere']
            noisy_labels = matdict['noisy_labels']
            expert_label_mask = matdict['expert_label_mask']
            # -1: Let class labels start at 0 in python
            labels = matdict['labels'] - 1

        expert_label_mask = expert_label_mask.astype(bool)
        icaact = icaweights @ icasphere @ data

        expert_label_mask = expert_label_mask.astype(bool)
        for ic_ind, ic in enumerate(icaact):
            time_idx = np.arange(0, ic.size - args.window_len + 1, args.window_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(args.window_len)[None, :]
            X[cum_ic_ind] = ic[time_idx]
            y[cum_ic_ind] = labels[ic_ind]
            noisy_labels_ar[cum_ic_ind] = noisy_labels[ic_ind]
            expert_label_mask_ar[cum_ic_ind] = expert_label_mask[ic_ind]
            subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

    return X, y, expert_label_mask_ar, subj_ind, noisy_labels_ar, labels


def load_raw_set_single_subj(args, rng, data_dir=Path('../data/cue'), fnames=None):
    #data_dir = Path(args.root, '/data/cue/')
    # the above line does not work on Caviness. I don't know why.
    # therefore,
    #data_dir = Path('../data/cue')
    #data_dir = Path('../data/codebooks/')


    #fnames = [f"subj-{i}.mat" for i in subj_ids] #modify to test on subset for smaller time

    #temp fix, change to all subjects later
    #fnames = fnames[0] #test a single subject for now
    file_list = [data_dir.joinpath(f) for f in fnames]
    # fnames = f'subj-{subj_id}.mat'
    # file_list = [data_dir.joinpath(fnames)]

    # print("data_dir: ", data_dir)
    # print("file_list:\n\t", file_list)

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=['labels', 'srate'])
            labels = matdict['labels']
            srate = matdict['srate']  # assumes all subjects have the same sampling rate
            srate = srate.item(0)  # `srate.shape=(1,1)`. This extracts the number.
            n_ics_per_subj.append(labels.shape[0])

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = (args.window_len / srate / 60)
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_win_per_ic, args.window_len), dtype=np.float32)
    y = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask_ar = np.full(n_ics, False)
    subj_ind = np.zeros(n_ics, dtype=int)
    # 7 ICLabel classes
    noisy_labels_ar = np.zeros((n_ics, 7), dtype=np.float32)
    #for file, subjID in tqdm(zip(file_list, args.subj_ids)):
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f)
            data = matdict['data']
            icaweights = matdict['icaweights']
            icasphere = matdict['icasphere']
            noisy_labels = matdict['noisy_labels']
            expert_label_mask = matdict['expert_label_mask']
            # -1: Let class labels start at 0 in python
            labels = matdict['labels'] - 1

        expert_label_mask = expert_label_mask.astype(bool)
        icaact = icaweights @ icasphere @ data

        expert_label_mask = expert_label_mask.astype(bool)
        for ic_ind, ic in enumerate(icaact):
            time_idx = np.arange(0, ic.size - args.window_len + 1, args.window_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(args.window_len)[None, :]
            X[cum_ic_ind] = ic[time_idx]
            y[cum_ic_ind] = labels[ic_ind]
            noisy_labels_ar[cum_ic_ind] = noisy_labels[ic_ind]
            expert_label_mask_ar[cum_ic_ind] = expert_label_mask[ic_ind]
            #subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

    return X, y, expert_label_mask_ar, noisy_labels_ar, labels #, subj_ind

def load_raw_set_single_subj_drb_frolich_extract(args, rng, data_dir=Path('../data/cue'), fnames=None):
    """
    So, I need the raw ICs to be in a specific shape for the BOWav feature extractor. Merely getting the raw ICs in a
    2d matrix is not enough, must be 3d tensor. Appropriate function above to work with different data structs in the
    saved mat file.
    Parameters
    ----------
    args
    rng
    data_dir
    fnames

    Returns
    -------

    """
    #data_dir = Path(args.root, '/data/cue/')
    # the above line does not work on Caviness. I don't know why.
    # therefore,
    #data_dir = Path('../data/cue')
    #data_dir = Path('../data/codebooks/')


    #fnames = [f"subj-{i}.mat" for i in subj_ids] #modify to test on subset for smaller time

    #temp fix, change to all subjects later
    #fnames = fnames[0] #test a single subject for now
    file_list = [data_dir.joinpath(f) for f in fnames]
    # fnames = f'subj-{subj_id}.mat'
    # file_list = [data_dir.joinpath(fnames)]

    print("data_dir: ", data_dir)
    print("file_list:\n\t", file_list)

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=['labels', 'srate'])
            labels = matdict['labels']
            #srate = matdict['srate']  # assumes all subjects have the same sampling rate
            #srate = srate.item(0)  # `srate.shape=(1,1)`. This extracts the number.
            n_ics_per_subj.append(labels.shape[0])

    srate = 256

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = (args.window_len / srate / 60)
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_win_per_ic, args.window_len), dtype=np.float32)
    y = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask_ar = np.full(n_ics, False)
    subj_ind = np.zeros(n_ics, dtype=int)
    # 6 classes
    #noisy_labels_ar = np.zeros((n_ics, 6), dtype=np.float32)
    #for file, subjID in tqdm(zip(file_list, args.subj_ids)):
    for file in tqdm(file_list):
        with file.open('rb') as f:
            matdict = loadmat(f)
            data = matdict['X']
            W = matdict['W']
            # data = matdict['data']
            # icaweights = matdict['icaweights']
            # icasphere = matdict['icasphere']
            #noisy_labels = matdict['noisy_labels']
            #expert_label_mask = matdict['expert_label_mask']
            # -1: Let class labels start at 0 in python
            labels = matdict['labels'] - 1

        #expert_label_mask = expert_label_mask.astype(bool)
        icaact = W @ data #icaweights @ icasphere @ data

        #expert_label_mask = expert_label_mask.astype(bool)
        for ic_ind, ic in enumerate(icaact):
            time_idx = np.arange(0, ic.size - args.window_len + 1, args.window_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(args.window_len)[None, :]
            X[cum_ic_ind] = ic[time_idx]
            y[cum_ic_ind] = labels[ic_ind]
            #noisy_labels_ar[cum_ic_ind] = noisy_labels[ic_ind]
            #expert_label_mask_ar[cum_ic_ind] = expert_label_mask[ic_ind]
            #subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

    return X, y, labels
    # return X, y, expert_label_mask_ar, subj_ind, noisy_labels_ar, labels