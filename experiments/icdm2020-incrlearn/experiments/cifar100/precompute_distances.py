from sklearn.metrics import pairwise_distances
import numpy as np


data = np.load('cifar_embeddings.npy')

nrows = ncols = data.shape[0]

dist = np.memmap('dist_memmapped.dat', dtype=np.float32,
                 mode='w+', shape=(nrows, ncols))

batch_size = 10000

for i in range(0, nrows, batch_size):
    print('Computing distances for batch {} to {}'.format(i, i + batch_size - 1))
    batch_dist = pairwise_distances(data[i:i + batch_size], data)
    dist[i:i + batch_size] = batch_dist
