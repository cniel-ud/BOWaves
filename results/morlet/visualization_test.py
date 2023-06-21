import numpy as np
import matplotlib.pyplot as plt


out_file = 'sikmeans_k-128_P-512_wlen-768.npz'

with np.load(out_file) as data:
    centroids = data['centroids']
    labels = data['labels']
    shifts = data['shifts']
    distances = data['distances']

unique_labels, cluster_size = np.unique(labels, return_counts=True)

# Sort centroids in descending order of cluster size
isort = np.argsort(-cluster_size)
centroids = centroids[isort]
unique_labels = unique_labels[isort]
cluster_size = cluster_size[isort]

#print(centroids)
#print(unique_labels)
#print(cluster_size)

#plt.plot(centroids)

# Set the figure size
#plt.figure(figsize=(10, 6))

# Iterate over the centroids and plot each as a waveform

#print(centroids[0])
'''
x=np.arange(len(centroids[0]))
plt.plot(x, centroids[0])

"""
for i, centroid in enumerate(centroids):
    # Generate x-axis values based on the length of the centroid
    x = np.arange(len(centroid))

    # Plot the waveform
    plt.plot(x, centroid, label=f"Centroid {i + 1}")
"""

# Set the plot title and labels
plt.title("Centroids as Waveforms")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Show a legend
#plt.legend()

# Display the plot
plt.show()
'''
# Assuming centroids contains the one-dimensional time series data

# Determine the grid dimensions based on the number of centroids

#determine number of centroids over some cluster size cutoff
cutoff = 5
number = 0
for i in cluster_size:
    if i >= 5:
        number += 1

num_centroids = len(centroids)
num_rows = int(np.ceil(np.sqrt(number)))
num_cols = int(np.ceil(number / num_rows))

# Create subplots with the determined grid dimensions
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Flatten the axs array if necessary
if num_centroids == 1:
    axs = np.array([axs])

# Iterate over the centroids and plot each as a waveform in a separate subplot
for i, centroid in enumerate(centroids):
    if cluster_size[i] >= 5:
        # Determine the subplot indices
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Plot the waveform in the corresponding subplot
        axs[row_idx, col_idx].plot(centroid)
        #axs[row_idx, col_idx].set_title(f"Centroid {i + 1}")
        axs[row_idx, col_idx].set_title(cluster_size[i])


# Remove empty subplots if the number of centroids is not a perfect square
if num_centroids % num_cols != 0:
    for i in range(num_centroids, num_rows * num_cols):
        axs.flatten()[i].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()