import pandas as pd
import random

dir="data/"

movie_review=pd.read_csv(dir+'Dataset_IMDB.csv')
fin_review=pd.read_csv(dir+'financialmarket_review.csv')
jet_patents=pd.read_csv(dir+'Jet-related Patents.csv')


randidx=random.sample(range(0,5006), 20)
movie_review=movie_review['Text'][randidx]
movie_review.to_csv(dir+'movie_review.csv')




