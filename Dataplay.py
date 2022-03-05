import datetime as dt
import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import re
import seaborn as sns
import nltk
from dateutil.relativedelta import relativedelta
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

from textblob import TextBlob

nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')

import datetime
from datetime import datetime as dt




def manipulate():
    df = pd.read_csv('',index_col=0)
    # print(df.time)
    three_months = dt.today() - relativedelta(months=+3)




    # # print(timeconvert(df.time[0]))
    # for i in df.time:
    #     df = df[~timeconvert2(df.time[i]) >= timeconvert1("2021-10-20")]
    #
    # df = df.time >= three_months
    # for i in df.time:
    #     if (df[timeconvert(df.time[i]) >= three_months]):
    #         df.drop(i)

    print(df)

def engagementCount():
    df = pd.read_csv('----', index_col=0)
    likes_list = df.likes.tolist()
    print("Average likes")
    print(likes_list)
    print("Sum of Likes")
    print(sum(likes_list))
    print("Max like on one post")
    print(max(likes_list))
    print("Minimum likes on a post")
    valueToBeRemoved = 0
    try:
        while True:
            likes_list.remove(valueToBeRemoved)
    except ValueError:
        pass

    print(min(likes_list))


    print("Retweets list:")
    retweets_list = df.retweets.tolist()
    print(retweets_list)
    print("Average Retweets")
    print(sum(retweets_list)/200)




def tweetfreq():
    df = pd.read_csv('', index_col=0)

    latest = df.time[0]
    last = df.time[199]

    print(last)
    print(latest)
    latest = (latest[0:10])
    print(latest)
    last = df.time[199]
    last = (last[0:10])
    print(last)

    latest_date = datetime.datetime.strptime(latest, '%Y-%m-%d')
    last_entry_date = datetime.datetime.strptime(last, '%Y-%m-%d')


    print(latest_date)
    time_dif = (latest_date - last_entry_date).days
    print(type(time_dif))

    print(200 / time_dif)


def collectReplies():

    jsonList = []
    replies_list = []
    with open('') as f:
        for jsonObj in f:
            jsonDict = json.loads(jsonObj)
            jsonList.append(jsonDict)

    for line in jsonList:
        for i in line["data"]:
            print(i['text'])
            replies_list.append(i['text'])
    print(len(replies_list))






        # print(f"result: {result}")
        # print(isinstance(result, dict))




def timeconvert(a):

    time = dt.strptime(a, "%Y-%m-%d")
    return time


def collectWordSearch():

    jsonList = []
    text_list =[]
    with open ('') as f:
        for jsonObj in f:
            jsonDict = json.loads(jsonObj)
            jsonList.append(jsonDict)


    # print(jsonList)
    for line in jsonList:
        for i in line["data"]:
            # print(i['text'])
            text_list.append(i['text'])

    ########################################################## SENTIMENTAL ANALYSIS #################################################


    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    number_of_tweets = len(text_list)
    pos_list = []
    neg_list = []
    neu_list = []


    for tweet in text_list:



        analysis = sid.polarity_scores(tweet)
        print(analysis)
        score = analysis['compound']

        if(score >= 0.05):
            positive +=1
            pos_list.append(tweet)
        elif(score <= -0.05):
            negative +=1
            neg_list.append(tweet)
        else:
            neutral += 1
            neu_list.append(tweet)

    positive = (positive/number_of_tweets)*100
    negative = (negative/number_of_tweets)*100
    neutral = (neutral/number_of_tweets)*100

    positive = format(positive, '.2f')
    neutral = format(neutral, '.2f')
    negative = format(negative, '.2f')

    # if(polarity == 0):
    #     print("Neutral")
    # elif(polarity <0):
    #     print("Negative")
    # elif (polarity > 0):
    #     print("Positive")

    # labels = ['Positive ['+str(positive)+'%]', 'Neutral [' +str(neutral)+'%]', 'Negative [' +str(negative) +'%]']
    # sizes = [positive,neutral,negative]
    # colors = ['yellowgreen', 'gold', 'red']
    # patches, texts = plt.pie(sizes, colors = colors, startangle=90)
    # plt.legend(patches, labels, loc ="best")
    # plt.title("  ")
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()



    # Top used Words

    df = pd.DataFrame({'tweets': neg_list})

    print(df)

    # Removing Retweets
    # df = df[~df.tweets.str.contains("RT")]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)



    print(df)

    list_of_sentences = [sentence for sentence in df.tweets]

    # Removing Punctuation

    lines = []
    for sentence in list_of_sentences:
        words = sentence.split()
        for w in words:
            lines.append(w)

    lines = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

    print(lines)

    lines2 = []

    for word in lines:
        if word != '':
            lines2.append(word)

    # Stemming the words to their root

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
    # df.plot(20)

    df = df[:30, ]
    plt.figure(figsize=(10, 5))
    sns.barplot(df.values, df.index, alpha=0.8)
    plt.title('Top Words Overall for ---- from Negative Replies')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.show()










collectReplies()
# collectWordSearch()
# manipulate()
# timeconvert()
# tweetfreq()
# engagementCount()

