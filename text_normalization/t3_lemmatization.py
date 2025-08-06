from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

### Download neccessary nltk modules (if not downloaded already)
# import nltk
# nltk.download('wordnet')

corpus="""\
Hello World! Welcome to the Sagar's nlp-plaground.
This playground has been created to play around with NLP concepts and for learning NLP practically.
"""

documents = sent_tokenize(corpus)

# Lemmatisation using WordNet Lemetizer with POS as noun
print( "\n\n","--"*30,"\n"," "*10 ,"Lemmatization with POS as Noun\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for word in words :
        print(lemmatizer.lemmatize(word, pos="n"), end=" ")

# Lemmatisation using WordNet Lemetizer with POS as verb
print( "\n\n","--"*30,"\n"," "*10 ,"Lemmatization with POS as Verb\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for word in words :
        print(lemmatizer.lemmatize(word, pos="v"), end=" ")

# Lemmatisation using WordNet Lemetizer with POS as adjective
print( "\n\n","--"*30,"\n"," "*10 ,"Lemmatization with POS as Adjective\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for word in words :
        print(lemmatizer.lemmatize(word, pos="a"), end=" ")

# Lemmatisation using WordNet Lemetizer with POS as adverb
print( "\n\n","--"*30,"\n"," "*10 ,"Lemmatization with POS as Adverb\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for word in words :
        print(lemmatizer.lemmatize(word, pos="r"), end=" ")


print("\n\n")