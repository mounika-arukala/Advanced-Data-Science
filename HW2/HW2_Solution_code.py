

import json
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


businessdata = []
with open('D:/mouni/business.json', "r",encoding='utf-8') as data1:
    for line in data1:
        businessdata.append(json.loads(line))
checkindata = []
with open('D:/mouni/checkin.json', "r",encoding='utf-8') as data1:
    for line in data1:
        checkindata.append(json.loads(line))

reviewdata = []
with open('D:/mouni/review.json', "r",encoding='utf-8') as data1:
    for line in data1:
        reviewdata.append(json.loads(line))

restaurant_id=[]
for r in businessdata:
    if((r['city']=="Pittsburgh" or r['city']=="Charlotte") and r['categories']!=None and "Restaurants" in r['categories'] ):
       restaurant_id.append(r['business_id'])

pittsburgh_restaurants=0
pittsburgh_non_restaurants=0       
for r in businessdata:
    if((r['city']=="Pittsburgh") and r['categories']!=None and "Restaurants" in r['categories'] ):
       pittsburgh_restaurants=pittsburgh_restaurants+1

for r in businessdata:
    if r['city']=="Pittsburgh":
        pittsburgh_non_restaurants=pittsburgh_non_restaurants+1 

       
charlotte_restaurants=0
charlotte_non_restaurants=0            
for r in businessdata:
    if((r['city']=="Charlotte") and r['categories']!=None and "Restaurants" in r['categories'] ):
        charlotte_restaurants=charlotte_restaurants+1
       
for r in businessdata:
    if r['city']=="Charlotte":
       charlotte_non_restaurants=charlotte_non_restaurants+1 
 
print(pittsburgh_restaurants)
print(pittsburgh_non_restaurants-pittsburgh_restaurants)
print(charlotte_restaurants)
print(charlotte_non_restaurants-charlotte_restaurants)
       
restaurant_checkin=[]    
no_checkin=[]    
for p in checkindata:
    if p['business_id'] in restaurant_id:
        restaurant_checkin.append([p['business_id'],len(p['time'])])
        no_checkin.append(len(p['time']))
        
average_checkin=sum(no_checkin)/len(no_checkin)        
restaurant_final=[]
for p in restaurant_checkin:
    if p[1]>(average_checkin):
        restaurant_final.append(p[0])

        
restaurant_review=[]
review_rating=[]
for p in businessdata:
    if p['business_id'] in restaurant_final:
        restaurant_review.append([p['business_id'],p['stars']])
        review_rating.append(p['stars'])

average_rating=sum(review_rating)/len(review_rating)        
restaurant_final=[]
for p in restaurant_review:
    if p[1]>(average_rating):
        restaurant_final.append(p[0])

types_pittsburgh=[]
for p in businessdata:
    if p['business_id'] in restaurant_final:
        if p['categories']!=None and p['city']=="Pittsburgh":    
            for type in p['categories']:
                types_pittsburgh.append(type)
                
pittsburgh_category=[]
for x in set(types_pittsburgh):
    count = types_pittsburgh.count(x)
    pittsburgh_category.append([count,x])
    
pittsburgh_category.sort(reverse=True)
print(pittsburgh_category)
                
types_charlotte=[]
for p in businessdata:
    if p['business_id'] in restaurant_final:
        if p['categories']!=None and p['city']=="Charlotte":    
            for type in p['categories']:
                types_charlotte.append(type)
                
charlotte_category=[]
for x in set(types_charlotte):
    count = types_charlotte.count(x)
    charlotte_category.append([count,x])
    
charlotte_category.sort(reverse=True)
print(charlotte_category)
        

crestaurant=[]
for p in businessdata:
    if p['categories']!=None:
        if "Chinese" in p['categories']:
            crestaurant.append(p['business_id'])

creview=[]
count=0            
for r in reviewdata:
    if r['business_id'] in crestaurant:
        creview.append(r['text'])
        
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()        
texts = []
for i in creview:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]    
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]   
    texts.append(stemmed_tokens)
dictionary = corpora.Dictionary(texts)    
cor = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(cor, num_topics=4, id2word = dictionary, passes=12)
print(ldamodel.print_topics(num_topics=4, num_words=6))

frequent=[]
for s in texts:
    for p in s:
        frequent.append(p)
q=Counter(frequent)
frequent_words=[(l,k) for k,l in sorted([(j,i) for i,j in q.items()], reverse=True)]             
dataframe_words=pd.DataFrame(frequent_words)
print(dataframe_words.head(10))
