from senticnet.senticnet import SenticNet
import nltk.data
from nltk.tokenize import wordpunct_tokenize
from nounadjranker import NJpairranker

def sentencepolarity(sentence):

    sum_pol=0
    toks=wordpunct_tokenize(sentence)
    for tok in toks:
        try:
            if tok in stop_words:
                toks.remove(tok)
                continue
            pol_intensity = float(sn.polarity_intense(tok))
        except Exception:
            continue
        sum_pol=sum_pol+pol_intensity

    #average polarity intensity for the sentence
    ave_pol=sum_pol/len(toks)
    #winner=max([posvote,negvote,neuvote])
    if ave_pol > 0:
            return "positive"
    elif ave_pol<0:
            return "negative"
    else:
        return "neutral"


def reviewmood(pairlist):
    moodlist=[]
    for pair in pairlist:
        try:
            mood = sn.moodtags(pair[0][1])
        except Exception:
            continue
        moodlist.append(mood)

    moodlist=[mood for sublist in moodlist for mood in sublist]
    return set(moodlist)

"""
def analyzesentics(sentence):

    senticdict={'aptitude':0,
    'sensitivity':0,
    'attention':0,
    'pleasantness':0}

    for tok in wordpunct_tokenize(sentence):
        try:
            senticdict['aptitude']+=float(sn.sentics(tok)['aptitude'])
            senticdict['sensitivity']+=float(sn.sentics(tok)['sensitivity'])
            senticdict['pleasantness']+=float(sn.sentics(tok)['pleasantness'])
            senticdict['attention']+=float(sn.sentics(tok)['attention'])
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
        return result"""

def reviewpolarity(pairlist):

    totalcount = sum([pair[1] for pair in pairlist])
    sum_pol = 0
    for pair in pairlist:
        adjtag = pair[0][1]
        try:
            pol_intensity = float(sn.polarity_intense(adjtag)) * pair[1] / totalcount
        except Exception:
            continue
        sum_pol = sum_pol + pol_intensity

    if sum_pol > 0:
            return "positive"
    elif sum_pol< 0:
            return "negative"
    else:
        return "neutral"


if __name__=="__main__":

    # stopwords
    stop_words = []
    with open('data/nltk_stopwords.txt', 'r') as f1:
        for line in f1.readlines():
            stop_words.append(line.rstrip('\n'))
        f1.close()

    print('--------Welcome to review sentiment analyzer!---------')

    while True:


        print('-------enter the review you wish to be analyzed:------')
        print ("      Enter/Paste your content+double Enter to save.     ")
        print("            or Ctrl+D to quit\n")

        review=''
        try:
            while True:
                line=input()
                if line:
                    review=review+line
                else:
                    break

        except EOFError:
            print('application terminted through keyboard interrupt')
            exit()

        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(review)
        # get noun-adj pairs
        pairlist = NJpairranker(sents, -1)
        #senticnet api
        sn=SenticNet()

        print('\n****analyzing polarity of the sentences...****')
        print()
        for sent in sents:
            print(('    The polarity of the sentence ###{}### is {}').format(sent, sentencepolarity(sent)))
        print()

        print('\n****analyzing polarity of the review...****')
        print()
        print(('    The polarity of the review is {}').format(reviewpolarity(pairlist)))

        print('****analyzing mood of this review....****')
        print()
        print('    Possible mood tags of this review:')
        for mood in reviewmood(pairlist):
            print(mood)
        print()
