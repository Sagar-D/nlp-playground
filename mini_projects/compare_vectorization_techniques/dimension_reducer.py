from sklearn.decomposition import PCA
import numpy as np

def reduce_to_2d(vectors) :
    """ Reduce dimensionality to 2d and remove outliers"""
    # Reduce dimensionality (PCA to 2D)
    print("Reducing Vectors to 2-D using PCA...")
    pca = PCA(n_components=2)
    reduced_bow_vector = pca.fit_transform(vectors)

    # Remove outliers
    # Compute bounds using percentiles and Filter points
    print("Removing outliers from the vector data...")
    x_low, x_high = np.percentile(reduced_bow_vector[:, 0], [1, 99])
    y_low, y_high = np.percentile(reduced_bow_vector[:, 1], [1, 99])
    reduced_bow_vector = reduced_bow_vector[
        (reduced_bow_vector[:, 0] >= x_low) & (reduced_bow_vector[:, 0] <= x_high) &
        (reduced_bow_vector[:, 1] >= y_low) & (reduced_bow_vector[:, 1] <= y_high)
    ]
    return reduced_bow_vector