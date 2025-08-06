from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

corpus="""\
Hello World! Welcome to the Sagar's nlp-plaground.
This playground has been created to play around with NLP concepts and for learning NLP practically.
"""

documents = sent_tokenize(corpus)

reduced_words = []
for sentence in documents :
    words = word_tokenize(sentence)
    for word in words :
        if word in stopwords.words() :
            print(f"Stop word found : {word}")
        else :
            reduced_words.append(word)

print("\n\n", "--"*30, " "*7, "\n Words after removing Stop Words \n", "--"*30, "\n" ,sep="")
print(" ".join(reduced_words))