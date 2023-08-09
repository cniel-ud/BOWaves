# Bag of Waves

This is a demo of the shift invariant k-means algorithm. Currently this is run on synthetic morlet data.

By default, the morlet data and results from running sikmeans on it are included in the repo.

If you'd like to run this on data of your own, create a subdirectory with the experiment name you'd like in both the data and results directories.
The data needs to be in .npz format. Then import the core API call from BOWaves.sikmeans.sikmeans_core as such:

    from BOWaves.sikmeans.sikmeans_core import run
    run('experiment_name', '/path/to/root/directory')

Additionally, if you want to run the code on some sample synthetic Morlet signal data:

    from BOWaves.sikmeans.sample_data_gen import generate_morlet_signal
    generate_morlet_signal()

If you've cloned the repo (use API when importing as package from pypi) and you'd like to still use the command line method to run the code, navigate to the scripts subdirectory in your terminal and run the following command:

    python run_sikmeans.py EXPERIMENT_NAME --root='root/directory'

There are other command line options available which you can find by checking run_sikmeans.py. 
By default, the script will save an image of the centroids it finds within your data that have at least 5 occurrences or more. This is a hyperparameter.

Note that if you are using the Bag-of-Waves feature extractor on EEG data, we currently assume that you have done some preprocessing
in order to extract the independent components (ICs), which is what the code will operate on.
This could be done with Matlab and EEGLAB, the MNE library in Python, etc.

Please let me (Austin Meek) know if you have any questions.

## Install

    pip install BOWaves


## Citation

Note: This repo is adapted from https://github.com/chmendoza/sikcsp. If you find it useful, please consider citing our paper:

```bibtex
@INPROCEEDINGS{9629913,
  author={Mendoza-Cardenas, Carlos H. and Brockmeier, Austin J.},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={Shift-invariant waveform learning on epileptic ECoG}, 
  year={2021},
  pages={1136-1139},
  doi={10.1109/EMBC46164.2021.9629913}}
```

## Further note

If you clone the repo in the current state, you may notice many things are in the repo that are not in the package, such as matlab files.
We are actively in the process of integrating MNE with our code to replace our older matlab data processing files.