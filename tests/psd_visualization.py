import matplotlib.pyplot as plt
from scipy.io import loadmat

"""
Want to visualize different data preprocessing of emotion & cue EEG datasets.
On plane, visualize frolich sampled and resampled. Wait until better internet connection to download and test emotion.
"""

title = 'frolich_extract_subj_03_resampled_to_mice_hp_filtered.mat'

frolich_resampled = loadmat(f'../data/codebooks/frolich/{title}')

pxx, freqs = plt.psd(frolich_resampled['X'], Fs=256)

#fig, axs = plt.subplots()

plt.title(title)

plt.show()

# arrange these later to be in a 3x4 grid etc

print()