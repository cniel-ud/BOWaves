"""
Integrate this into the utilities module a bit later, once cmmn integration is stable and tested.

Additionally, since these will be rather large tensors, how can I reshape them in place?
Potentially I'll use einsum for this.

For now, assume that the data is stored in an npz. Although for the purposes of the package, I'll integrate
capabilities for loading the data from .mat files as well since that's what EEG software typically puts out.
"""

import numpy as np
from typing import List

def load_tensor(directory_path: str, file_list: List[str], srate: int, segment_length: int) -> np.ndarray:
    """
    This function is for loading in the tensor data for each subject. Usage:
    Provide the directory path and the list of files to load.
    For now, the files should be in .npz or .npy format.

    Crucially, we assume that ICA has already been performed on the data and that
    we can load the ICs from the 'icaact' key.
    The ICs should be of the shape (n_ics, length).

    Here we treat each IC as a separate source domain for the CMMN multi-source domain adaptation.
    For more info, read the docs (add docs later).

    :param directory_path: The directory to the path containing the files.
    :param file_list: A list of the files from which to load. Include the filetype at the end.
    :param srate: The sampling rate of the data.
    :param segment_length: The length of the segments to use for PSD estimation for the CMMN.
    :return: A tensor of shape (n_ics, n_signals, time=30 seconds).
    """

    # Load the data from the files.
    data = [np.load(directory_path + file) for file in file_list]

    window_length = srate * 30 # 30 second windows
    n_signals = segment_length // window_length
    tensor = np.ndarray(shape=(0, n_signals, window_length))

    for subject in data:
        # Check that the data has the 'icaact' key.
        if 'icaact' not in subject.keys():
            raise ValueError("The data does not have the 'icaact' key.")

        ics, ic_length = subject['icaact'].shape

        # reshape to (n_ics, n_signals, time=30 seconds)

        # curtail the data to the correct length.
        curtailed = subject['icaact'][:, :n_signals*window_length]

        # make sure to test that this reshaping does it correctly.
        reshaped_data = curtailed.reshape(ics, n_signals, window_length)

        # concatenate the reshaped data to the tensor.
        tensor = np.concatenate((tensor, reshaped_data), axis=0)

    return tensor