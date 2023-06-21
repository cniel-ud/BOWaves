# shift_kmeans demo

This is a demo of the shift invariant k-means algorithm. Currently this is run on synthetic morlet data.

By default, the morlet data and results from running sikmeans on it are included in the repo.

If you'd like to run this on data of your own, create a subdirectory with the experiment name you'd like in both the data and results directories.
The data needs to be in .npz format. Then, navigate to the scripts subdirectory in your terminal and run the following command:

python run_sikmeans.py EXPERIMENT_NAME --root='root/directory'

There are other command line options available which you can find by checking run_sikmeans.py. 
By default the script will save an image of the centroids it finds within your data that have at least 5 occurrences or more. This is a hyperparameter.

Please let me (Austin Meek) know if you have any questions.

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
