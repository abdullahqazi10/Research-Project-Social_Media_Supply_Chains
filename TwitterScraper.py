

import datetime as dt
import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import re
import seaborn as sns
import nltk
from dateutil.relativedelta import relativedelta

nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from datetime import date, timedelta







api_key = " "
api_secret_key = ""
bearer_token = ""
access_token = ""
access_token_secret=""




def setup():
    auth = tweepy.OAuthHandler(api_key,api_secret_key)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth)

    number_of_tweets = 200

    tweets = []
    likes = []
    time = []
    media=[]
    retweets = []



    for i in tweepy.Cursor(api.user_timeline,
                           screen_name='',
                           tweet_mode="extended",
                           exclude_replies=True,
                           include_rts=True).items(number_of_tweets):

            # items(number_of_tweets):
        tweets.append(i.full_text)
        likes.append(i.favorite_count)
        time.append(i.created_at)
        retweets.append(i.retweet_count)
        media.append(i.entities)

    df = pd.DataFrame({'tweets': tweets, 'likes': likes, 'time': time, 'retweets':retweets, 'media': media})

    # print(df)


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    df.to_csv('')

    # print(df)

    list_of_sentences = [sentence for sentence in df.tweets]

    #Removing Punctuation

    lines = []
    for sentence in list_of_sentences:
        words = sentence.split()
        for w in words:
            lines.append(w)

    lines = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

    print(lines)

    # Tokenization
    # Words are extracted from the sentences
    lines2 = []

    for word in lines:
        if word != '':
            lines2.append(word)

    #Stemming the words to their root

    from nltk.stem.snowball import SnowballStemmer

    # The Snowball Stemmer requires that you pass a language parameter
    s_stemmer = SnowballStemmer(language='english')

    stem = []
    for word in lines2:
        stem.append(s_stemmer.stem(word))

    # Removing all Stop Words

    stem2 = []

    for word in stem:
        if word not in stopwords.words():
            stem2.append(word)

    df = pd.DataFrame(stem2)

    df = df[0].value_counts()

    # df
    # df['freq'] = df.groupby(0)[0].transform('count')
    # df['freq'] = df.groupby(0)[0].transform('count')
    # df.sort_values(by = ('freq'), ascending=False)

    # This will give frequencies of our words

    from nltk.probability import FreqDist

    freqdoctor = FreqDist()

    for words in df:
        freqdoctor[words] += 1

    # This is a simple plot that shows the top 30 words being used


    df = df[:30, ]
    plt.figure(figsize=(10, 5))
    sns.barplot(df.values, df.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.show()






# setup()
