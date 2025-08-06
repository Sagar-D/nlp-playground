from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

corpus="""\
Hello World! Welcome to the Sagar's nlp-plaground.
This playground has been created to play around with NLP concepts and learn\
NLP practically.
"""

documents = sent_tokenize(corpus)

# Stemming using Porter Stemmer
print("--"*30, "\n"," "*10 ,"Porter Stemmer\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    porter_stemmer = PorterStemmer()
    for word in words :
        print(porter_stemmer.stem(word) , end=" ")


# Stemming using Snowball Stemmer
print( "\n\n","--"*30,"\n"," "*10 ,"Snowball Stemmer\n","--"*30, sep="")
for sentence in documents :
    words = word_tokenize(sentence)
    snowball_stemmer = SnowballStemmer("english")
    for word in words :
        print(snowball_stemmer.stem(word), end=" ")

print("\n\n")
