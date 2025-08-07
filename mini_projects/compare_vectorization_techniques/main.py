import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# personal modules
from plot_graph import plot_graph
from text_pre_processor import clean_text, pre_process_text
from vectorizer import bag_of_words_vectorization
from data_processor import reduce_to_2d, remove_outliers

TRAINING_DATA_SIZE = 3000
PROJECT_PATH = "mini_projects/compare_vectorization_techniques/"
TRAINING_DATA_FOLDER = PROJECT_PATH + "training_data/"
OUTPUT_PLOT_FOLDER = PROJECT_PATH + "output/plots/"


### Read Training data and normalize text ###

email_docs = pd.read_csv(TRAINING_DATA_FOLDER + "email.csv", header=0)
email_docs = email_docs[:TRAINING_DATA_SIZE]
email_docs["tokens"] = email_docs["Message"].apply(clean_text).apply(pre_process_text)
email_docs["processed_message"] = email_docs["tokens"].apply(
    lambda token_list: " ".join(token_list)
)

### Bag of Words Vectorization with ngram (1,3) ###

print("Initiating Bag of Words Vectorization with ngram (1,3)")
bow_vectors = bag_of_words_vectorization(
    email_docs["processed_message"], ngram_range=(1, 3)
)

print("Reducing NoW vectors to 2 dimension")
bow_vectors = reduce_to_2d(bow_vectors)

# Add spam/ham labels to vector data
bow_vectors = np.column_stack((bow_vectors, email_docs["Category"]))
bow_vectors = remove_outliers(bow_vectors)

print("Plotting graph of BoW Vectors...")
plot_graph(bow_vectors, file_path=OUTPUT_PLOT_FOLDER + "bow_1_3_plot.png")


### Bag of Words Vectorization with ngram (2,3) ###

print("Initiating Bag of Words Vectorization with ngram of (2,3)")
bow_vectors = bag_of_words_vectorization(
    email_docs["processed_message"], ngram_range=(2, 3)
)

print("Reducing NoW vectors to 2 dimension")
bow_vectors = reduce_to_2d(bow_vectors)

# Add spam/ham labels to vector data
bow_vectors = np.column_stack((bow_vectors, email_docs["Category"]))
bow_vectors = remove_outliers(bow_vectors, min_percentile=5, max_percentile=95)

print("Plotting graph of BoW Vectors...")
plot_graph(bow_vectors, file_path=OUTPUT_PLOT_FOLDER + "bow_2_3_plot.png")
