# Perform all the neccessary text preprocessing techinques on the a text corpus
# and extract only neccessary data
# Step 1 : Remove punctuations
# Step 2 : Remove special characters / emojis
# Step 3 : Expand contractions
# Step 4 : Spelling correction
# Step 4 : Tokenisation
# Step 5 : lower casing
# Step 6 : POS tagging
# Step 7 : Lemmetization
# Step 8 : Remove Stop Words

import string
import emoji
import contractions
from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk

corpus = """\
"OMG!!! I can't beleive it's already 10:30 p.m... 😱 Time's flyin' fast, isn't it? \
Anyway, I've gotta get bak to work—haven’t finished my assgnmnt yet. \
Ugh, this is soooo stressfull!! 🙄💻 But hey, atleast I’m tryin’, right? lol! 🤷‍♂️"
"""

# Remove punctuations
corpus = corpus.translate(corpus.maketrans("","",string.punctuation))

# Expand Contractions
corpus = contractions.fix(corpus)

# Remove emojis
corpus = emoji.demojize(corpus)

# Tokenize to sentences
documents = sent_tokenize(corpus)

# Tokenize to words and lower casing
words = []
for sentence in documents :
    words.extend([word.lower() for word in word_tokenize(sentence)])

# Spelling correction
spellChecker = SpellChecker()
spell_corrected_words = []
for word in words :
    spell_corrected_words.append(spellChecker.correction(word) if spellChecker.correction(word) != None else word)
words = spell_corrected_words

# Create a POS tagging for tokens
pos_tagged_words = pos_tag(words)

def get_wordnet_pos_tag(treebank__pos_tag):
    """Method to convert Treebank POS tags to WordNetLemmatizer specific tags"""
    if treebank__pos_tag.startswith('J'):
        return "a"
    elif treebank__pos_tag.startswith('V'):
        return "v"
    elif treebank__pos_tag.startswith('N'):
        return "n"
    elif treebank__pos_tag.startswith('R'):
        return "r"
    else:
        return "n"

# Lemmatize the words using the POS tags
lemmetizer = WordNetLemmatizer()
root_words = [ lemmetizer.lemmatize(word, get_wordnet_pos_tag(tag)) for word, tag in pos_tagged_words ]

# Remove stop words
stop_words = set(stopwords.words("english"))
pre_processed_words = [word for word in root_words if word not in stop_words]

# Final preprocessed words
print(" ".join(pre_processed_words))