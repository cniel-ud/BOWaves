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

