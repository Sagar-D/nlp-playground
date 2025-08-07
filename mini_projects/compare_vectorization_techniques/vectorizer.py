from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

NOW_MAX_FEATURES = 500


## Implement Bag of Words Vectorization

def bag_of_words_vectorization(
    processed_documents: List[str],
    max_features: int = NOW_MAX_FEATURES,
    binary: bool = False,
    ngram_range: Tuple[int, int] = (1, 3),
):
    """Perform BoW vectorization on a list of pre_processed documents"""
    # Bag of words Vectorization
    print("Performing Bag of Words Vectorization...")
    bow_vectorizer = CountVectorizer(
        max_features=max_features, binary=binary, ngram_range=ngram_range
    )
    bow_vector = bow_vectorizer.fit_transform(processed_documents).toarray()
    return bow_vector
