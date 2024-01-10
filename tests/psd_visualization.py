import matplotlib.pyplot as plt
from scipy.io import loadmat

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
plt.savefig('../data/codebooks/frolich/frolich_500_hz_psd_no_filtering.png')

# arrange these later to be in a 3x4 grid etc

print()