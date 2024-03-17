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

    # Extract the ICs from the data.
    ics = [subject['icaact'] for subject in data]

    # Reshape the ICs to be of shape (n_ics, n_signals, time=30 seconds).
    ics = [ic.reshape(-1, srate*30) for ic in ics]

    # Stack the ICs into a tensor.
    tensor = np.stack(ics, axis=0)

    return tensor