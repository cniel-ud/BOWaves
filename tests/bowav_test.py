"""
Test for BoWav feature extractor.

Get ICs from sample data, then test what BoWav finds. If it makes sense, pass the test.

"""

from pathlib import Path
import numpy as np

#change to your root directory to run the test. Hardcoded here since this is a simple sanity check, not included in the package
root = '/home/austin/PycharmProjects/BOWaves'

root = Path(root)
data_dir = root.joinpath('data', 'morlet')
data_dir.mkdir(exist_ok=True)

results_dir = root.joinpath('results')
results_dir.mkdir(exist_ok=True)

results_dir = root.joinpath('results', 'morlet')
results_dir.mkdir(exist_ok=True)

fpath = list(data_dir.glob('*.npz'))[0]
with np.load(fpath, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

print(T.shape)