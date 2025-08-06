from nltk import pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize, word_tokenize

# import nltk
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

corpus = """\
Hello World! Welcome to the Sagar's nlp-plaground.
I studied in MS Ramaiah Institute of Technology.
I have worked in multiple companies including Practo Technologies and Walmart Global Tech.
"""

documents = sent_tokenize(corpus)

words = []
for sentence in documents:
    words.extend(word_tokenize(sentence))


tagged_words = pos_tag(words)
named_identity_words = ne_chunk(tagged_words)
named_identity_words.draw()
