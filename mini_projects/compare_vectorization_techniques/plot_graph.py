import matplotlib.pyplot as plt
from matplotlib.patches import mpatches
from typing import List


def plot_graph(vector_data: List[List[float, float, str]]):
    """Plot scatter graph for vectors using matplotlib"""

    plt.figure(figsize=(8, 6))
    COLOR_MAP = {"spam": "red", "ham": "green"}

    for i, vector in enumerate(vector_data):
        plt.scatter(
            vector[0],
            vector[1],
            color=COLOR_MAP[vector[2]],
            label=vector[2],
            s=5,
            alpha=0.7,
        )

    plt.title("PCA Projection of BoW Vectors")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)

    # Create custom legend
    custom_legend = [
        mpatches.Patch(color="green", label="Ham email"),
        mpatches.Patch(color="red", label="Spam email"),
    ]
    plt.legend(handles=custom_legend, title="")
    plt.show()
