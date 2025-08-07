import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def clean_text(text: str):
    """Romove punctuations,extra spaces and handle contractions, cases from text"""

    # Remove punctuations
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s", " ", text)
    # Lower case the text
    text = text.lower()
    # Fix contractions
    text = contractions.fix(text)
    return text


def pre_process_text(text: str):
    """Tokkenize, Lemmatize and remove Stopwords and return List of Tokens"""

    words = word_tokenize(text)
    pos_tagged_words = pos_tag(words)

    def get_wordnet_pos_tag(treebank_tag: str):
        if treebank_tag.startswith("J"):
            return "a"
        elif treebank_tag.startswith("V"):
            return "v"
        elif treebank_tag.startswith("R"):
            return "r"
        else:
            return "n"

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    root_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos_tag(tag))
        for word, tag in pos_tagged_words
        if word not in stop_words
    ]
    return root_words
