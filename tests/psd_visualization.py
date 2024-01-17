#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os

from BOWaves.utilities import dataloaders

#%%
"""
Want to visualize different data preprocessing of emotion & cue EEG datasets.
On plane, visualize frolich sampled and resampled. Wait until better internet connection to download and test emotion.
"""

# title = 'frolich_extract_subj_03_resampled_to_mice_hp_filtered.mat'
#
# frolich_resampled = loadmat(f'../data/codebooks/frolich/{title}')
#
# pxx, freqs = plt.psd(frolich_resampled['X'], Fs=256)



# Create subplots in a 3x4 grid
fig, axs = plt.subplots(3, 4, figsize=(15, 10))

# Flatten the 3x4 subplot array to make it easier to index
axs = axs.flatten()


num_signals = 12
# Plot each signal
for i in range(num_signals):
    # Load data
    if i < 9:
        title = f'frolich_extract_subj_0{i+1}_500_hz_lp_filtered.mat'#_500_hz_hp_filtered.mat'
    else:
        title = f'frolich_extract_subj_{i+1}_500_hz_lp_filtered.mat'#_500_hz_hp_filtered.mat'
    #title = f'frolich_extract_subj_{i+1}_resampled_to_mice_hp_filtered.mat'
    cue = loadmat(f'../data/codebooks/frolich/{title}')

    #pxx, freqs = plt.psd(frolich_resampled['X'], Fs=256)
    pxx, freqs = axs[i].psd(cue['X'], Fs=500)
    axs[i].set_title(f'Subject {i + 1}')

# Add a title to the entire figure
fig.suptitle('PSD of Cue 500 hz EEG Data 1 hz Low Pass Filtered', fontsize=16)

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])

#fig, axs = plt.subplots()

#plt.title(title)

#plt.show()
plt.savefig('../data/codebooks/frolich/frolich_500_hz_psd_lp_filtered.png')

# arrange these later to be in a 3x4 grid etc

print()

#%%

#In this cell, visualize each PSD by label. Load all data,
# then apply ICA. For each channel, sort into specific labels.
# then plot PSD for each label

file_list = []
# load data
num_signals = 12
# Plot each signal
for i in range(num_signals):
    # Load data
    if i < 9:
        title = f'frolich_extract_subj_0{i+1}_500_hz_hp_filtered.mat'#_500_hz_hp_filtered.mat'
    else:
        title = f'frolich_extract_subj_{i+1}_500_hz_hp_filtered.mat'#_500_hz_hp_filtered.mat'
    #title = f'frolich_extract_subj_{i+1}_resampled_to_mice_hp_filtered.mat'
    file = f'./data/codebooks/frolich/{title}'
    file_list.append(file)

print(os.getcwd())

# hold classified ICs. Technically here we only want the brain ones, but this is boilerplate at the moment
neural = {'name': 'neural', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
blink = {'name': 'blink', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
muscle = {'name': 'muscle', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
mixed = {'name': 'mixed', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
lateyes = {'name': 'lateyes', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
heart = {'name': 'heart', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}

all_classes = [neural, blink, muscle, mixed, lateyes, heart]

for file in file_list:
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

# now visualize each class of ICs
for class_ in all_classes:
    # take 12 random samples from each class
    num_samples = 12
    sample_indices = np.random.randint(0, len(class_['ICs']), num_samples)

    # get samples from ICs
    samples = []
    for index in sample_indices:
        samples.append(class_['ICs'][index])

    # plot each sample
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    axs = axs.flatten()
    for i, sample in enumerate(samples):
        pxx, freqs = axs[i].psd(sample, Fs=500)
        axs[i].set_title(f'Sample {i + 1}')

    # Add a title to the entire figure
    fig.suptitle(f'PSD of {class_["name"]} ICs', fontsize=16)

    # save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'../data/codebooks/frolich/{class_["name"]}_500_hz_psd_hp_filtered.png')