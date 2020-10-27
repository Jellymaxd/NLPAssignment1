import nltk.data
import seaborn as sns
import matplotlib.pyplot as plt
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


"""-----------------------------sentence segmentation----------------------------------"""
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sent_fin=sent_detector.tokenize(fin_doc)
sent_jet=sent_detector.tokenize(jet_doc)
sent_movie=sent_detector.tokenize(movie_doc)

with open(dir+'sentences/sentences_fin.txt', 'w', encoding='utf-8') as f1:
    for sent in sent_fin:
        f1.write('--'+sent+'--')
    f1.close()

with open(dir+'sentences/sentences_jet.txt', 'w', encoding='utf-8') as f2:
    for sent in sent_jet:
        f2.write('--'+sent+'--')
    f2.close()

with open(dir+'sentences/sentences_movie.txt', 'w', encoding='utf-8') as f3:
    for sent in sent_movie:
        f3.write('--'+sent+'--')
    f3.close()


#get len of tokens
len_finsent=[len(sent.split(' ')) for sent in sent_fin]
len_jetsent=[len(sent.split(' ')) for sent in sent_jet]
len_moviesent=[len(sent.split(' ')) for sent in sent_movie]

fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(16,20) )
fig.suptitle('distribution of sentence length in financial review(left), jet related patents(middle) and movie review(right)', fontsize=16)

sns.distplot(len_finsent, kde=False, hist=True, ax=axes[0])
axes[0].set_xlabel('sentence length', fontsize=14)
axes[0].set_ylabel('number of sentences', fontsize=14)
sns.distplot(len_jetsent, kde=False, hist=True,ax=axes[1])
axes[1].set_xlabel('sentence length', fontsize=14)
axes[1].set_ylabel('number of sentences', fontsize=14)
sns.distplot(len_moviesent, kde=False, hist=True, ax=axes[2])
axes[2].set_xlabel('sentence length', fontsize=14)
axes[2].set_ylabel('number of sentences', fontsize=14)
plt.show()


"""---------------------------------------pos tagging-------------------------------------"""
fin_sentence="Stock market linkage or segmentation is an important factor when estimating the impact of liberalizations on the cost of capital through an international asset-pricing model with investment restrictions."
jet_sentences="The struts are supported downstream of a last row of rotating turbine blades."
movie_sentences="None of the actors in THE SECRET OF ROAN INISH are likely to be familiar to American viewers."
print(nltk.pos_tag(nltk.word_tokenize(fin_sentence)))
print(nltk.pos_tag(nltk.word_tokenize(jet_sentences)))
print(nltk.pos_tag(nltk.word_tokenize(movie_sentences)))

