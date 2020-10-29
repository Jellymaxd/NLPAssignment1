import nltk.data
import pandas as pd
import spacy
from nltk.stem.porter import *



"""
noun_adj_pairs={}
for i in range(len(postags)-1):
    pair=(postags[i][0], postags[i+1][0])
    if (pair in noun_adj_pairs):
        noun_adj_pairs[pair] = noun_adj_pairs[pair]+1
    else:
        if ('NN' in postags[i][1] and 'JJ' in postags[i+1][1]):
            noun_adj_pairs[pair]=1
        else:
            continue

noun_adj_pairs={k: v for k, v in sorted(noun_adj_pairs.items(), key=lambda item: item[1])}
print(noun_adj_pairs)
"""
def NJpairranker(sentlist, topk):
    noun_adj_pair={}

    # stopwords
    stop_words = []
    with open('data/nltk_stopwords.txt', 'r') as f1:
        for line in f1.readlines():
            stop_words.append(line.rstrip('\n'))
        f1.close()

    #using spacy dependency parsing
    nlp = spacy.load("en_core_web_sm")
    for sent in sentlist:
        doc = nlp(sent)
        for token in doc:
            #print(token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])

            #amod
            if token.dep_=='amod':
                #key exist, update
                if (token.head.text, token.text) in noun_adj_pair:
                    noun_adj_pair[(token.head.text, token.text)] = noun_adj_pair[(token.head.text, token.text)]+1
                else:
                    noun_adj_pair[(token.head.text, token.text)]=1

            #acomp

            elif token.head.pos_=='AUX' or 'VERB' in token.head.pos_:
                childofAUX=[child for child in token.children if child.dep_=='nsubj' or child.dep_=='acomp']
                childdep=[child.dep_ for child in childofAUX]
                childofAUX = [child.text for child in childofAUX]

                #does not have both nsubj and acomp
                if (('acomp' not in childdep) or ('nsubj' not in childdep)):
                    continue

                #key exist,update
                elif tuple(childofAUX) in noun_adj_pair:
                    noun_adj_pair[tuple(childofAUX)]=noun_adj_pair[tuple(childofAUX)]+1
                #create key
                else:
                    noun_adj_pair[tuple(childofAUX)]=1

            else:
                continue

    #remove stop words
    noun_adj_pair={k: v for k, v in noun_adj_pair.items() if k[0] not in stop_words}
    #final dict with NN-JJ pair: frequency
    noun_adj_pair={k: v for k, v in sorted(noun_adj_pair.items(), key=lambda item: item[1], reverse=True)}

    """with stemmed nouns"""
    stemmer=PorterStemmer()
    stemmed_pairs={}
    for k,v in noun_adj_pair.items():
        tempkey=stemmer.stem(k[0])
        if tempkey in stemmed_pairs:
            stemmed_pairs[tempkey].append((k[1],v))
        else:
            stemmed_pairs[tempkey]=[(k[1],v)]

    #final dict with NN-JJ pairs with frequency (NN stemmed)
    stemmed_pairs={k: v for k, v in sorted(stemmed_pairs.items(), key=lambda item: len(item[1]), reverse=True)}
    stemmed_pairs =list(stemmed_pairs.items())
    #return (list((stemmed_pairs).items())[:5])
    resultlist=list((noun_adj_pair).items())
    return (resultlist[0:topk] if len(resultlist)>=topk else resultlist)



if __name__=="__main__":

    dir = "data/"

    # review csv
    pes_review = pd.read_csv(dir + 'pes2021_steamreview.csv')

    # concat reviews
    pes_doc = ''
    for doc in (pes_review['Text']):
        pes_doc = pes_doc + doc + ' '

    # segment into sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_pes = sent_detector.tokenize(pes_doc)
    print('noun-adjective pair | frequency')
    for pair in NJpairranker(sent_pes,5):
        print(pair)

    #postags = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sent_pes]


