"""
So we are testing two different types of resampling.
Firstly, we want to take the emotion codebooks and resample them to 500 hz.
Secondly, we want to take the cue data and resample (down sample) them to 256 hz.
This is for the second method. First method is in resample_emotion_to_cue.py

To do this, first load the ICs and labels. Then resample the ICs, and save them to cue/cue_signals_resampled_to_emotion/.
When I load the data, it's from a mat file. So I want to save it to mat file as well, along with all the peripherals
such as sampling rate (srate, changed), labels, etc.


"""

import scipy #from here use resample, loadmat and savemat
import numpy as np
import pyrootutils
from pathlib import Path

pyrootutils.set_root(path='/work/cniel/ajmeek/BOWaves/BOWaves', pythonpath=True)

data_dir = Path('../data/cue')

subj_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
fnames = [f"subj-{i}.mat" for i in subj_ids]
file_list = [data_dir.joinpath(f) for f in fnames]

for file in file_list:
    matdict = scipy.io.loadmat(file)

    # would like to just resample the data arrays and change the srate to 256.
    # but, if it's only valid to change the ICs (icaweights @ icasphere @ data), then
    # I need to do that inside the dataloader when it calculates that.
    # perhaps I can just change the srate here and have conditional logic in the dataloader,
    # but I'd much prefer to simply resample the data and srate fields here and leave that code alone.

    # for scipne to resample axis 1. but it's 1979155 right now. So do (500/256) * 1979155 and put that
    # in as the num parameter in scipy's resample method.
    # First though, talk with Isabel and ask her to verify the documentation since idk signals processing

    new_centroids = scipy.signal.resample(centroids, 500, axis=1)

# TODO - this is missing the resample. not sure when I had taken that out