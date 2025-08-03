from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from pprint import pprint


corpus="""\
Hello World! Welcome to the Sagar's nlp-plaground.
This playground has been created to play around with NLP concepts and learn\
NLP practically.
"""

# tokenise corpus (paragraph) to documents (sentences)
documents = sent_tokenize(corpus)
print(f"\n{len(documents)} Sentence Tokens created: \n\n{documents}\n")

# tokenise documents (sentences) to words
print("\n\nWord Tokenisation\n")
for document in documents :
    print(word_tokenize(document))
print("\n")

# Tokenise words using wordpunct_tokenize => considers all punctuations as a seperate token
print("\n\nWord Tokenisation with wordpunct_tokenize\n")
for document in documents :
    print(wordpunct_tokenize(document))
print("\n")