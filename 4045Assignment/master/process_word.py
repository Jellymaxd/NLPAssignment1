from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import *
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

dir="data/"

movie_review=pd.read_csv(dir+'movie_review.csv')
fin_review=pd.read_csv(dir+'financialmarket_review.csv')
jet_patents=pd.read_csv(dir+'Jet-related Patents.csv')

movie_doc=jet_doc=fin_doc=""

for doc in (fin_review.iloc[:,1]):
    fin_doc=fin_doc+doc+' '

for doc in (jet_patents.iloc[:,1]):
    jet_doc=jet_doc+doc+' '

for doc in (movie_review['Text']):
    movie_doc=movie_doc+doc+' '

"""--------------------tokenize----------------------"""

tokens_fin=set(wordpunct_tokenize(fin_doc))
tokens_jet=set(wordpunct_tokenize(jet_doc))
tokens_movie=set(wordpunct_tokenize(movie_doc))

with open(dir+'tokenfiles/tokens_fin.txt', 'w', encoding='utf-8') as f1:
    for tok in tokens_fin:
        f1.write(tok+' ')
    f1.close()

with open(dir+'tokenfiles/tokens_jet.txt', 'w',encoding='utf-8') as f2:
    for tok in tokens_jet:
        f2.write(tok+' ')
    f2.close()

with open(dir+'tokenfiles/tokens_movie.txt', 'w',encoding='utf-8') as f3:
    for tok in tokens_movie:
        f3.write(tok+' ')
    f3.close()


"""--------------------stemming----------------------"""

stemmer=SnowballStemmer("english")
stemmed_fin=set([stemmer.stem(fintok) for fintok in tokens_fin])
stemmed_jet=set([stemmer.stem(jettok) for jettok in tokens_jet])
stemmed_movie=set([stemmer.stem(movietok) for movietok in tokens_movie])

with open(dir+'stemmedtokens/stemmed_fin.txt', 'w', encoding='utf-8') as f1:
    for tok in stemmed_fin:
        f1.write(tok+' ')
    f1.close()

with open(dir+'stemmedtokens/stemmed_jet.txt', 'w', encoding='utf-8') as f2:
    for tok in stemmed_jet:
        f2.write(tok+' ')
    f2.close()

with open(dir+'stemmedtokens/stemmed_movie.txt', 'w', encoding='utf-8') as f3:
    for tok in stemmed_movie:
        f3.write(tok+' ')
    f3.close()


#get len of tokens
len_fin_unstemmed=[len(tok) for tok in tokens_fin]
len_fin_stemmed=[len(tok) for tok in stemmed_fin]
len_jet_unstemmed=[len(tok) for tok in tokens_jet]
len_jet_stemmed=[len(tok) for tok in stemmed_jet]
len_movie_unstemmed=[len(tok) for tok in tokens_movie]
len_movie_stemmed=[len(tok) for tok in stemmed_movie]


"""--------------------compare tokens metrics before and after stemming----------------------"""
print('number of unique tokens\n')
print(' '*20+'unstemmed:'+10*' '+'stemmed')
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('financial_review: ', len(tokens_fin),len(stemmed_fin)))
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('jet_patents: ',len(tokens_jet),len(stemmed_jet)))
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('movie_review: ',len(tokens_movie),len(stemmed_movie)))
print('\n')

print('max and min\n')
print(' '*20+'unstemmed:'+10*' '+'stemmed')
print(' '*20+'max '+' min'+10*' '+'max '+'min ')
print(('{:<20}{:<2}'+3*' '+'{:<2}'+10*' '+'{:<2}'+3*' '+'{:<2}').format('financial_review: ', max(len_fin_unstemmed),min(len_fin_unstemmed),
                                             max(len_fin_stemmed),min(len_fin_stemmed)))
print(('{:<20}{:<2}'+3*' '+'{:<2}'+10*' '+'{:<2}'+3*' '+'{:<2}').format('jet_patents: ',max(len_jet_unstemmed),min(len_jet_unstemmed),
                                             max(len_jet_stemmed),min(len_jet_stemmed)))
print(('{:<20}{:<2}'+3*' '+'{:<2}'+10*' '+'{:<2}'+3*' '+'{:<2}').format('movie_review: ',max(len_movie_unstemmed),min(len_movie_unstemmed),
                                             max(len_movie_stemmed),min(len_movie_stemmed)))

print('\n')
print('average token length\n')
print(' '*20+'unstemmed:'+10*' '+'stemmed')
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('financial_review: ', sum(len_fin_unstemmed)/len(len_fin_unstemmed),
                                             sum(len_fin_stemmed)/len(len_fin_stemmed)))
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('jet_patents: ',sum(len_jet_unstemmed)/len(len_jet_unstemmed),
                                             sum(len_jet_stemmed)/len(len_jet_stemmed)))
print(('{:<20}{:<9}'+10*' '+'{:<9}' ).format('movie_review: ',sum(len_movie_unstemmed)/len(len_movie_unstemmed),
                                             sum(len_movie_stemmed)/len(len_movie_stemmed)))


""""--------------------distribution of token length before and after stemming----------------------"""



fig, axes = plt.subplots(nrows=3,ncols=2, figsize=(16,20))
fig.suptitle('distribution of token length before(left) and after(right) stemming', fontsize=16)

sns.distplot(len_fin_unstemmed, kde=True, hist=True, ax=axes[0][0]).set_title('unstemmed financial review')
sns.distplot(len_fin_stemmed, kde=True, hist=True,ax=axes[0][1]).set_title('stemmed financial review')
sns.distplot(len_jet_unstemmed, kde=True, hist=True, ax=axes[1][0]).set_title('unstemmed jet patent')
sns.distplot(len_jet_stemmed, kde=True, hist=True,ax=axes[1][1]).set_title('stemmed jet patent')
sns.distplot(len_movie_unstemmed, kde=True, hist=True, ax=axes[2][0]).set_title('unstemmed movie review')
sns.distplot(len_movie_stemmed, kde=True, hist=True, ax=axes[2][1]).set_title('stemmed movie review')

plt.tight_layout(pad=10.0)
plt.show()