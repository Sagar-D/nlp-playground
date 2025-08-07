import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Union


def plot_graph(vector_data: List[List[Union[float, float, str]]], file_path=""):
    """Plot scatter graph for vectors using matplotlib"""

    plt.figure(figsize=(12, 7))
    COLOR_MAP = {"spam": "red", "ham": "green"}

    for i, vector in enumerate(vector_data):
        plt.scatter(
            vector[0],
            vector[1],
            color=COLOR_MAP[vector[2]],
            label=vector[2],
            s=7,
            alpha=0.7,
        )

    plt.title("PCA Projection of BoW Vectors")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)

    # Create custom legend
    custom_legend = [
        Patch(color="green", label="Ham email"),
        Patch(color="red", label="Spam email"),
    ]
    plt.legend(handles=custom_legend, title="")

    # Save image to filepath
    if file_path.strip() != "" :
        try:
            plt.savefig(file_path)
            print(f"Graph Plot saved to location : {file_path}")
        except Exception as error:
            print(f"Failed to save the plot to file location : {file_path}. Error: {error}")
            print(str(error))
    plt.show()
