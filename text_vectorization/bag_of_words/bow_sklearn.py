from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import contractions
from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

df = pd.read_csv("text_vectorization/training_data/email.csv", header=0)
DATA_SET_SIZE = 50000
df = df[:DATA_SET_SIZE]


def clean_text(text):
    # Remove punctuations
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Lower case text
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    return text


def spell_correction(tokens):
    """Spell corrections"""
    spell_checker = SpellChecker()
    spell_corrected_words = []
    for token in tokens:
        corrected_word = spell_checker.correction(token)
        if corrected_word != None:
            spell_corrected_words.append(corrected_word)
    return spell_corrected_words


def get_wordnet_pos_tag(treebank__pos_tag):
    """Method to convert Treebank POS tags to WordNetLemmatizer specific tags"""
    if treebank__pos_tag.startswith("J"):
        return "a"
    elif treebank__pos_tag.startswith("V"):
        return "v"
    elif treebank__pos_tag.startswith("N"):
        return "n"
    elif treebank__pos_tag.startswith("R"):
        return "r"
    else:
        return "n"


def lemmatize_tokens(tokens):

    lemmatizer = WordNetLemmatizer()
    pos_tagged_tokens = pos_tag(tokens=tokens)
    return [
        lemmatizer.lemmatize(token, get_wordnet_pos_tag(tag))
        for token, tag in pos_tagged_tokens
    ]


def remove_stop_words(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]


df["Message"] = df["Message"].apply(clean_text)
df["tokens"] = df["Message"].apply(word_tokenize)
# df["tokens"] = df["tokens"].apply(spell_correction) Ignore spell check to reduce time
df["tokens"] = df["tokens"].apply(lemmatize_tokens)
df["processed_messages"] = df["tokens"].apply(lambda tokens: " ".join(tokens))


print("\n" + " " * 15 + "Sample Processed Data : \n")
print(df["processed_messages"].head())
print("\n\n")

print("--" * 30)
print(" " * 15 + "Bag of Words Vectorization")
print("--" * 30)
bow_vectorizer = CountVectorizer(max_features=500)
X = bow_vectorizer.fit_transform(df["Message"])
print(
    f"\nNo of Features (Words) selected :  {len(bow_vectorizer.get_feature_names_out())}"
)
print(f"\n\nFeature Vector : \n{X.toarray()}")
