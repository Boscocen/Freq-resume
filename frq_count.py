import nltk
import re
import pandas as pd 
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.collocations import *

from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
#from geotext import GeoText

#import spacy
#python -m spacy download en_core_web_sm

resumeDS = pd.read_csv('resume_dataset.csv', encoding='utf-8')
#print out job title and numbers of resume associated to the job title
print (resumeDS['Category'].value_counts())

def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

sentences = resumeDS['Resume'].values
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append("months")
monthstopwords = ["january", "february", "march", "april", "may", "june", "july", 
"august", "september", "octorber", "november", "december"]
stopwords.append(monthstopwords)

totalResumeWords = []

for i in range (0, len(sentences)):
	removeSpace = re.sub(' +', ' ',sentences[i])
	text = removeSpace
	tokenizeText = nltk.word_tokenize(text)
	for w in tokenizeText:
		if w.lower() not in stopwords and not alpha_filter(w):
			totalResumeWords.append(w)

lowerResumeWords = [w.lower() for w in totalResumeWords]
#lemmatize words
lemmatizeWords = [lemmatizer.lemmatize(w) for w in lowerResumeWords]
wordFreq = FreqDist(lemmatizeWords)
mostCommon = wordFreq.most_common(20)

spell = Speller(lang='en')
mostCommon_correct = [(spell(w),t) for w,t in mostCommon]

for item in mostCommon_correct:
	print (item)

top20words = [w[0] for w in mostCommon_correct]
resumebigrams = list(nltk.bigrams(top20words))

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(top20words)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)


