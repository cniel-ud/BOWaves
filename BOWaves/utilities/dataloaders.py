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


def load_and_visualize_mat_file_frolich(file_path, up_to=None, visualize=False):
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

    # Visualize EEG time series data
    X = data['X'] #raw
    W = data['W'] #demixing matrix

    Y = W @ X #combine here to get the ICs

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
        labels_in_order.append(data['classes'][0][i-1][0][0])

    return Y, labels_in_order

# Replace 'your_file.mat' with the actual file path
#Y, labels = load_and_visualize_mat_file_frolich('../../data/frolich/frolich_extract_01.mat')#, visualize=True)

#print()