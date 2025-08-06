from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

### Download neccessary nltk modules (if not downloaded already)
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

corpus="""\
Hello World! Welcome to the Sagar's nlp-plaground.
This playground has been created to play around with NLP concepts and for learning NLP practically.
"""

documents = sent_tokenize(corpus)

for sentence in documents :
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    for pos_tag_item in pos_tags :
        print(pos_tag_item)

