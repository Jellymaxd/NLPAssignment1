from senticnet.senticnet import SenticNet
import nltk.data
import spacy
from nltk.stem.porter import *

def NJpairranker(sentlist, topk):
    noun_adj_pair={}


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
            elif token.pos_=='AUX':
                childofAUX=[child for child in token.children if child.dep_=='nsubj' or child.dep_=='acomp']
                childdep=[child.dep_ for child in childofAUX]
                childofAUX = [child.text for child in childofAUX]
                #does not have both nsubj and acomp
                if ('acomp' not in childdep or 'nsubj' not in childdep):
                    continue

                #key exist,update
                elif tuple(childofAUX) in noun_adj_pair:
                    noun_adj_pair[tuple(childofAUX)]=noun_adj_pair[tuple(childofAUX)]+1
                else:
                    noun_adj_pair[tuple(childofAUX)]=1

            else:
                continue

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

    #return (list((stemmed_pairs).items())[:5])
    resultlist=list((noun_adj_pair).items())
    return (resultlist[0:topk] if len(resultlist)>=topk else resultlist)


def analyzepolarity(NJpairs):

    totalcount = sum([pair[1] for pair in NJpairs])
    posvote=negvote=neuvote=0
    sum_pol=0
    for pair in NJpairs:
        adjtag = pair[0][1]
        try:
            pol_intensity = float(sn.polarity_intense(adjtag))*pair[1]/totalcount
        except Exception:
            continue
        sum_pol=sum_pol+pol_intensity

        try:
            pol_value=sn.polarity_value(adjtag)
            if pol_value=='positive':
                posvote=posvote+pair[1]
            elif pol_value=='negative':
                negvote=negvote+pair[1]
            else:
                neuvote=neuvote+pair[1]
        except Exception:
            continue

    winner=max([posvote,negvote,neuvote])
    if posvote==winner:
        if sum_pol>0.4:
            return "strongly positive"
        elif sum_pol>0.2:
            return "weakly positive"
    elif negvote == winner:
        if sum_pol < -0.4:
            return "strongly negative"
        elif sum_pol < -0.2:
            return "weakly negative"
    else:
        return "neutral"

def analyzemood(NJpairs):
    moodlist=[]
    for pair in NJpairs:
        adjtag = pair[0][1]
        try:
            mood = sn.moodtags(adjtag)
        except Exception:
            continue
        moodlist.append(mood)

    moodlist=[mood for sublist in moodlist for mood in sublist]
    return set(moodlist)

def analyzesentics(NJpairs):
    totalcount = sum([pair[1] for pair in NJpairs])

    senticdict={'aptitude':0,
    'sensitivity':0,
    'attention':0,
    'pleasantness':0}

    for pair in NJpairs:
        try:
            adjtag = pair[0][1]
            senticdict['aptitude']=senticdict['aptitude']+float(sn.sentics(adjtag)['aptitude'])*pair[1]/totalcount
            senticdict['sensitivity'] = senticdict['sensitivity'] + float(sn.sentics(adjtag)['sensitivity'])*pair[1]/totalcount
            senticdict['pleasantness'] = senticdict['pleasantness'] + float(sn.sentics(adjtag)['pleasantness'])*pair[1]/totalcount
            senticdict['attention'] = senticdict['attention'] + float(sn.sentics(adjtag)['attention'])*pair[1]/totalcount
        except Exception:
            continue
    return(hourglass_emotions(senticdict))

def hourglass_emotions(senticdict):
    emolist=[]
    for k, v in senticdict.items():
        pl = at = sen = apt = -1
        if k == 'pleasantness':
            if v > 0:
                pl = 1
            elif v<0:
                pl= 0
        elif k == 'attention':
            if v > 0:
                at = 1
            elif v<0:
                at= 0
        elif k == 'sensitivity':
            if v > 0:
                sen = 1
            elif v<0:
                sen= 0
        elif k == 'aptitude':
            if v > 0:
                apt = 1
            elif v<0:
                apt = 0

        hourglass_dict={
        (1, 1, 1, 1): ('optimistic', 'aggressive', 'love', 'rivalry'),
        (1, 1, 1, 0): ('optimistic', 'aggressive', 'gloat', 'contempt'),
        (1, 1, 0, 1): ('jokingly', 'rejection', 'love', 'rivalry'),
        (1, 1, 0, 0):('jokingly', 'rejection', 'gloat', 'contempt'),
        (1, 0, 1, 1): ('anxious', 'optimistic', 'submissive', 'love'),
        (1, 0, 0, 1):('jokingly', 'awe', 'love', 'rivalry'),
        (1, 0, 0, 0): ('optimistic', 'aggressive', 'gloat', 'coercive'),
        (0, 1, 1, 1):('frustrated', 'aggressive', 'envious', 'rivalry'),
        (0, 1, 1, 0): ('optimistic', 'aggressive', 'remorse', 'contempt'),
        (0, 1, 0, 1):('disapproval', 'rejection', 'envious', 'rivalry'),
        (0, 1, 0, 0): ('disapproval', 'rejection', 'remorse', 'contempt'),
        (0, 0, 1, 1): ('frustrated', 'anxious', 'envious', 'submissive'),
        (0, 0, 1, 0): ('frustrated', 'anxious', 'remorse', 'contempt'),
        (0, 0, 0, 1): ('disapproval', 'awe', 'envious', 'submissive'),
        (0, 0, 0, 0): ('disapproval', 'awe', 'remorse', 'coercive')}

        result=hourglass_dict[pl, sen, at, apt] if (pl, sen, at, apt) in hourglass_dict else ('neutral')
        return result



if __name__=="__main__":

    print('--------Welcome to review sentiment analyzer!---------')

    while True:


        print('--------enter the review you wish to be analyzed:-------')
        print ("Enter/Paste your content+double Enter to save.")
        print("or enter q+double Enter to quit\n")

        review=''
        while True:
            line = input()
            review = review + line
            if line.strip() == '':
                break

        if review=='q' or review=='Q':
            exit()

        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(review)

        # get noun-adj pairs
        pairlist = NJpairranker(sents, -1)
        sn = SenticNet()


        print('\n****analyzing polarity of this review....****')
        print()
        print('    The polarity of this review is: ',analyzepolarity(pairlist))
        print()

        print('****analyzing mood of this review....****')
        print()
        print('    Possible mood tags of this review:')
        for mood in analyzemood(pairlist):
            print(mood)
        print()

        print('****analyzing emotions of this review....****')
        emolist=analyzesentics(pairlist)
        print()
        print('    Possible emotions of this review:')
        print(emolist)
        print()
