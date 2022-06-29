import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

df = pd.read_excel(r'C:\Users\user\Desktop\bot kt\dialog_talk_bot.xlsx')
df.ffill(axis=0, inplace = True)
df1 = df.head(10)
def step1(a):
    for x in a:
        b = str(x).lower()
        p = re.sub(r'[^а-я0-9]', ' ', b)
        print(p)
step1(df1['Context'])
nltk.data.find(r'C:\Users\user\AppData\Roaming\nltk_data\tokenizers\punkt\russian.pickle')
s = 'Расскажи о себе'
words = word_tokenize(s)
print(words)

import pymorphy2
morph = pymorphy2.MorphAnalyzer()
for i in words:
    p = morph.parse(i)[0]
    print(i,'-', p.tag.POS)


text = words

def lemmatize(text):

    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return res

print(lemmatize(text))

import nltk
#nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

#Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

#Preprocess function
def text_normalization(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text
text_normalization(s)

df['Lemmatized'] = df['Context'].apply(text_normalization)
df.head(20)

#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('russian')
print(stop)

cv = CountVectorizer()
X = cv.fit_transform(df['Lemmatized']).toarray()

features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)
df_bow.head(20)

Quest = 'Ты бесишь меня каждым символом своего кода'
Q = []
a = Quest.split()
for i in a:
    if i in stop:
        continue
    else:
        Q.append(i)
    b = ' '.join(Q)
print (b)    

Quest_lemma = text_normalization(b)
Quest_bow = cv.transform([Quest_lemma]).toarray()
print(Quest_bow)

cosine_value = 1 - pairwise_distances(df_bow, Quest_bow, metric = 'cosine')
print(cosine_value)
df['Simil_bow'] = cosine_value
df.head()
df_simi = pd.DataFrame(df, columns = ['Text Response', 'Simil_bow'])
df_simi.head()

df_simi_sort = df_simi.sort_values(by = 'Simil_bow', ascending = False)
df_simi_sort.head()

threshold = 0.2
df_tres = df_simi_sort[df_simi_sort['Simil_bow'] > threshold]
df_tres.tail()

index_val = cosine_value.argmax()
df['Text Response'].loc[index_val]

Quest1 = 'Сколько тебе лет'
tfidf = TfidfVectorizer()
Quest1_lemma = text_normalization(Quest1)
print(Quest1_lemma)
tfidf.fit([Quest1_lemma])
#Quest1_tfidf = tfidf.transform([Quest1_lemma]).toarray()
Quest1_tfidf = tfidf.transform([Quest1_lemma]).toarray()
print(Quest1_tfidf)

tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(df['Lemmatized']).toarray()
df_tfidf = pd.DataFrame(x_tfidf, columns = tfidf.get_feature_names())
df_tfidf.head(10)
print(df['Lemmatized'])

cos = 1 - pairwise_distances(df_tfidf, Quest1_tfidf, metric = 'cosine')
print(cos)

df['sim_tfidf'] = cos
df_simi_tfidf = pd.DataFrame(df, columns = ['Text Response', 'sim_tfidf'])
df_simi_tfidf.head()

df_simi_tfidf_sort = df_simi_tfidf.sort_values(by='sim_tfidf', ascending = False)
df_simi_tfidf_sort.head()

df_tresh = df_simi_tfidf_sort[df_simi_tfidf_sort['sim_tfidf'] > threshold]
df_tresh.head()

index_value = cos.argmax()
print(Quest1)
print(df['Text Response'].loc[index_value])
df['Context'].loc[index_value]
