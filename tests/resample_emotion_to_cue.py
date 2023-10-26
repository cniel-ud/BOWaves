"""
This will be a short script to resample emotion study codebooks to cue sampling rate.

Located in /work/cniel/ajmeek/BOWaves/BOWaves/data/codebooks

The codebooks are saved as .npz files using the following command:
for label in labels:
    np.savez(out_file, centroids=label['centroids'], labels=label['labels'],
                         shifts=label['shifts'], distances=label['distances'], inertia=label['inertia'])

I only care about the learned centroids for the BOW algo so that's the only thing I'll resample.

"""

import scipy
import numpy as np

# load codebook
#codebook = np.load('/work/cniel/ajmeek/BOWaves/BOWaves/data/codebooks/emotion/sikmeans_P-256_k-16_class-1_minutesPerIC-50_icsPerSubj-2.npz')

for i in range(7):
    codebook = np.load(f'/work/cniel/ajmeek/BOWaves/BOWaves/data/codebooks/emotion/sikmeans_P-256_k-128_class-{i+1}_minutesPerIC-50_icsPerSubj-2.npz')
    centroids = codebook['centroids']


    # resample centroids
    # currently emotion sampling rate is 256 hz. this should work.
    new_centroids = scipy.signal.resample(centroids, 500, axis=1)


    # save new centroids
    # do I need to specifically save all the other things?
    np.savez(f'/work/cniel/ajmeek/BOWaves/BOWaves/data/codebooks/emotion_resampled_to_cue/sikmeans_P-256_k-128_class-{i+1}_minutesPerIC-50_icsPerSubj-2_resampled.npz', centroids=new_centroids)