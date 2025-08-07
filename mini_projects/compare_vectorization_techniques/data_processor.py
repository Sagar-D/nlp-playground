from sklearn.decomposition import PCA
import numpy as np


def reduce_to_2d(vectors):
    """Reduce dimensionality to 2d and remove outliers"""
    # Reduce dimensionality (PCA to 2D)
    print("Reducing Vectors to 2-D using PCA...")
    pca = PCA(n_components=2)
    reduced_bow_vector = pca.fit_transform(vectors)
    return reduced_bow_vector


def remove_outliers(vectors, min_percentile=1, max_percentile=99):
    """Removing outliers from the vector data"""

    # Compute bounds using percentiles and Filter vector points
    print("Removing outliers from the vector data...")
    x_low, x_high = np.percentile(vectors[:, 0], [min_percentile, max_percentile])
    y_low, y_high = np.percentile(vectors[:, 1], [min_percentile, max_percentile])
    processed_vectors = vectors[
        (vectors[:, 0] >= x_low)
        & (vectors[:, 0] <= x_high)
        & (vectors[:, 1] >= y_low)
        & (vectors[:, 1] <= y_high)
    ]
    print(f"Selected vectors size : {len(processed_vectors)}")
    return processed_vectors
