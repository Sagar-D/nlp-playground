import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# personal modules
from plot_graph import plot_graph
from text_pre_processor import clean_text, pre_process_text
from vectorizer import bag_of_words_vectorization
from dimension_reducer import reduce_to_2d

TRAINING_DATA_SIZE = 3000


### Read Training data and normalize text ###

email_docs = pd.read_csv(
    "mini_projects/compare_vectorization_techniques/training_data/email.csv", header=0
)
email_docs = email_docs[:TRAINING_DATA_SIZE]
email_docs["tokens"] = email_docs["Message"].apply(clean_text).apply(pre_process_text)
email_docs["processed_message"] = email_docs["tokens"].apply(
    lambda token_list: " ".join(token_list)
)


### Bag of Words Vectorization" ###

print("Initiating Bag of Words Vectorization")
bow_vectors = bag_of_words_vectorization(email_docs["processed_message"])
print("Reducing NoW vectors to 2 dimension")
reduced_bow_vector = reduce_to_2d(bow_vectors)
# Add spam/ham labels to vector data
reduced_bow_vector_with_labels = np.column_stack(
    (reduced_bow_vector, email_docs["Category"])
)
print("Plotting graph of BoW Vectors...")
plot_graph(reduced_bow_vector_with_labels)
