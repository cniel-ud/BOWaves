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
2. Load ICs, load from matlab file outputs using sklearn, use sklearn for train test split
3. Dictionary and codebook creation files for BOWaves. Associated dataloaders
"""

from scipy.io import loadmat
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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


def load_and_visualize_mat_file_frolich(file_path):
    # Create 'img' subdirectory if it doesn't exist
    img_dir = os.path.join(os.path.dirname(file_path), 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Load .mat file
    data = loadmat(file_path)

    # Display metadata
    print("Metadata:")
    for key, value in data.items():
        if not key.startswith("__"):
            print(f"{key}: {type(value)}")
            # if isinstance(value, np.ndarray):
            #     print(value)

    # Visualize EEG time series data
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64, np.int32, np.int64):
            if value.ndim == 2:
                num_channels, num_samples = value.shape
                time = np.arange(num_samples)  # Assuming time starts from 0 and is evenly spaced

                fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))
                fig.suptitle(key)

                for channel in range(num_channels):
                    axes[channel].plot(time, value[channel])
                    #axes[channel].set_ylabel(f'Channel {channel + 1}')
                    if not (key == "X" and channel == 63):
                        print(channel)
                        axes[channel].set_ylabel(f'Channel {channel} \n Label {data["classes"][0][data["labels"][channel] - 1][0][0]}')

                axes[-1].set_xlabel('Time')
                plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

                # Save the plot to the 'img' subdirectory
                plot_filename = os.path.join(img_dir, f"{key}_plot.png")
                plt.savefig(plot_filename)
                plt.close()

            else:
                print(f"Skipped visualization for {key} due to unsupported shape.")

    plt.show()

    #print(data['classes'][0][0])

    for count, i in enumerate(data['labels']):
        print(f"IC #{count} is label {data['classes'][0][i-1][0][0]}")

# Replace 'your_file.mat' with the actual file path
load_and_visualize_mat_file_frolich('../../data/frolich/frolich_extract_01.mat')