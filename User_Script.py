from turtle import bgcolor
import streamlit as st
from os.path import exists
import import_ipynb
import functions as fnc
import import_ipynb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud
import requests
import shutil
import functions as fnc
from absl import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, OrderedDict
import cv2 as cv
import os
from itertools import chain
from collections import Counter, OrderedDict
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
import spacy
from streamlit_tags import st_tags
import networkx as nx
import csv
from pyvis.network import Network
import streamlit.components.v1 as components
from stvis import pv_static


def convert_df(df):
     return df.to_csv()

def get_count(lst, word):
    cnt = 0
    for x in lst:
        if word.lower() in x:
            cnt+=1
    return cnt

st.set_page_config(layout="wide")
st.title('Twitter Users Analysis')
idxs = st_tags(
    label='## Enter Author Ids:',
    text='Press enter to add more',
    maxtags=100)
for i in range(len(idxs)):
    idxs[i] = int(idxs[i])

filepath = './datasets/Users_data_'
timeline_path = filepath+'timeline'
authors_path = filepath+'info'

if st.button('Load Authors Data'):
    
    print('getting Authors data')
    with st.spinner('Getting Authors Data...'):
        fnc.get_user_info(filename=authors_path, user_ids=idxs, user_field=['created_at','protected','verified', 'public_metrics'])
    st.success('Done!!')

    authors = pd.read_csv(authors_path+'.csv', names=['id','created_at', 'name', 'username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))    

    authors.id = authors.id.astype(str)
    st.download_button(
        label="Download Filtered Authors Data as CSV",
        data=convert_df(authors),
        file_name='auhtors_info.csv',
        mime='text/csv',
    )

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.scatterplot(data = authors, x='username', y='short_date')
    plt.ylabel('creation date')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='tweets_count')
    plt.ylabel('Total Number of Tweets')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='followers')
    plt.ylabel('Total Number of followers')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='following')
    plt.ylabel('Total Number user following')
    plt.xlabel('author_id')
    st.pyplot(fig)


    for idx in idxs:
        followers_filepath = './datasets/'+str(idx)+'_followers.csv'
        following_filepath = './datasets/'+str(idx)+'_following.csv'

        if not exists(followers_filepath):
            fnc.get_followers(idx, followers_filepath)

        if not exists(following_filepath):
            fnc.get_following(idx, following_filepath)

    for idx in idxs:
        id_following = pd.read_csv('./datasets/'+str(idx)+'_following.csv', names=['following_id', 'following_username', 'following_name'])
        id_followers = pd.read_csv('./datasets/'+str(idx)+'_followers.csv', names=['follower_id', 'follower_username', 'follower_name'])
        st.download_button(
            label="Download Followers of "+authors[authors.id == str(idx)]['username'].values[0],
            data=convert_df(id_followers),
            file_name=authors[authors.id == str(idx)]['username'].values[0]+'_followers.csv',
            mime='text/csv',
        )
        st.download_button(
            label="Download Following of "+authors[authors.id == str(idx)]['username'].values[0],
            data=convert_df(id_following),
            file_name=authors[authors.id == str(idx)]['username'].values[0]+'_following.csv',
            mime='text/csv',
        )

    # Creating Network Graph
    G = nx.Graph()
    cnt = 0

    for idx in idxs:
        G.add_node(str(idx),color='red', title=authors[authors.id == str(idx)]['username'].values[0])

    followers = {}
    following = {}

    for idx in idxs:
        id_following = pd.read_csv('./datasets/'+str(idx)+'_following.csv', names=['following_id', 'following_username', 'following_name'])
        id_followers = pd.read_csv('./datasets/'+str(idx)+'_followers.csv', names=['follower_id', 'follower_username', 'follower_name'])
        followers[idx] = id_followers
        following[idx] = id_following

    for idx in idxs:
        for node,username in zip(following[idx].following_id, following[idx].following_username):
            lst = []
            for idx2 in idxs:
                if node in followers[idx2].follower_id:
                    lst.append(idx2)
                elif node in following[idx2].following_id:
                    lst.append(idx2)
            if len(lst) > 1:
                if not G.has_node(str(node)):
                        G.add_node(str(node),color='blue', title=username)
                for source in lst:
                    G.add_edge(str(source), str(node))

        for node,username in zip(followers[idx].follower_id, followers[idx].follower_username):
            lst = []
            for idx2 in idxs:
                if node in followers[idx2].follower_id:
                    lst.append(idx2)
                elif node in following[idx2].following_id:
                    lst.append(idx2)
            if len(lst) > 1:
                if not G.has_node(str(node)):
                        G.add_node(str(node),color='blue', title=username)
                for source in lst:
                    G.add_edge(str(source), str(node))

    st.header('Network Graph of Users')
    net = Network("1000px", "2000px",notebook=True, font_color='#10000000')
    net.from_nx(G)
    pv_static(net)
    # 
    # nx.draw(G, with_labels=True)
    # 
    # net.show('Network_Graph.html')
    # HtmlFile = open("Network_Graph.html", 'r', encoding='utf-8')
    # source_code = HtmlFile.read() 
    # components.html(source_code, height = 1200,width=1000)



limit = st.number_input('Enter Numbr Of Tweets for Analysis', min_value=1, format='%i')
word = st.text_input('Enter word For Sentiment Analysis')

if st.button('Timelines Analysis'):

    
    print('Getting timelines')
    with st.spinner('Getting Timeline Data...'):
        fnc.get_timeline(timeline_path, idxs ,datetime.datetime.now(),['id','created_at', 'public_metrics', 'source', 'lang'], lim=limit)
    st.success('Done!!')
    
    authors = pd.read_csv(authors_path+'.csv', names=['id','created_at', 'name', 'username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))

    timelines = pd.read_csv(timeline_path+'.csv', names = ['author_id', 'tweet_id','source', 'tweet_time', 'text', 'lang', 'likes', 'retwets'], parse_dates=['tweet_time'])
    timelines['created_at'] = timelines['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0])

    st.download_button(
        label="Download Timelines of Authors as CSV",
        data=convert_df(timelines),
        file_name='authors_timelines.csv',
        mime='text/csv',
    )
    nolang = ['qme',  'und','qst', 'qht',  'zxx', 'qam']
    dataX_nottext = timelines[timelines.lang.isin(nolang)]
    dataX_text = timelines[~timelines.lang.isin(nolang)]
    dataX_text['author_username'] = dataX_text.author_id.apply(lambda x : authors[authors.id == x]['username'].values[0])

    with st.spinner('Text Pre Processing...'):
        translated = []
        for x,lang in zip(dataX_text.text, dataX_text.lang):
            if lang != 'en':
                translated.append(fnc.get_translate(x))
            else :
                translated.append(x)
        dataX_text['text'] = translated
        dataX_text['text'] = dataX_text['text'].apply(lambda x : fnc.clean_text(x))
        dataX_text['text'] = dataX_text['text'].apply(lambda x : fnc.clean_stopWords(x))
        dataX_text['text_tokens'] = dataX_text['text'].apply(lambda x : fnc.tokenize(x))
        dataX_text['text_lemmatized'] = dataX_text['text_tokens'].apply(lambda x : fnc.lemmatize(x))
        dataX_text['sentiment'] = dataX_text['text_lemmatized'].apply(lambda x : fnc.get_sentiment(x))
    st.success('Done!!')

    st.download_button(
        label="Download authors non text tweets as CSV",
        data=convert_df(dataX_nottext),
        file_name='authors_nontext.csv',
        mime='text/csv',
    )
    
    for id,username in zip(authors.id, authors.username):
        st.subheader('Frequent Words Used by author - '+username)
        fig = plt.figure(figsize=(15,5))
        negative = dataX_text[dataX_text.author_id == id].text_tokens
        negative = [" ".join(negative.values[i]) for i in range(len(negative))]
        negative = [" ".join(negative)][0]
        wc = WordCloud(min_font_size=3,max_words=200,width=1600,height=720, colormap = 'Set1', background_color='black').generate(negative)
        plt.imshow(wc,interpolation='bilinear')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        st.pyplot(fig)

    fig = plt.figure(figsize=(25,4))
    st.header('counts of neutral, negtive and positive tweets for authors')
    dataX_text.author_id = dataX_text.author_id.astype(str)
    plt.xticks(rotation=90)
    sns.histplot(data=dataX_text, x='author_username', hue=dataX_text.sentiment, multiple='dodge', palette='bright', discrete=True, shrink=.9)
    st.pyplot(fig)
    dataX_text.author_id = dataX_text.author_id.astype(int)

    author_sentiments = {'username' : [], 'positive tweets' : [], 'neutral tweets' : [], 'negative tweets': []}

    for idx in dataX_text.author_id.unique():
        author_sentiments['username'].append(authors[authors.id == idx]['username'].values[0])
        for ind, va in zip(dataX_text[dataX_text.author_id == idx].sentiment.value_counts().index, dataX_text[dataX_text.author_id == idx].sentiment.value_counts().values):
            if ind == 'Positive':
                author_sentiments['positive tweets'].append(va)
            elif ind == 'Negative':
                author_sentiments['negative tweets'].append(va)
            else:
                author_sentiments['neutral tweets'].append(va)

        if len(author_sentiments['positive tweets']) < len(author_sentiments['username']):
            author_sentiments['positive tweets'].append(0)
        if len(author_sentiments['negative tweets']) < len(author_sentiments['username']):
            author_sentiments['negative tweets'].append(0)
        if len(author_sentiments['neutral tweets']) < len(author_sentiments['username']):
            author_sentiments['neutral tweets'].append(0)

    sentiment_df = pd.DataFrame(author_sentiments)
    st.download_button(
        label="Download All Sentiment Counts of Authors as CSV",
        data=convert_df(sentiment_df),
        file_name='authors_sentiments.csv',
        mime='text/csv',
    )

    dataX_text['word_present'] = dataX_text.text_tokens.apply(lambda x : get_count(x,word))
    sent = {}
    cnt = 0
    for idx,x,sen in zip(dataX_text.author_username, dataX_text.word_present, dataX_text.sentiment):
        if sen == 'Positive' and x == 1:
            sen = 1
        elif sen == 'Negative' and x == 1:
            sen = -1
        else:
            sen = 0
        if idx in sent:
            sent[idx] += sen
        else:
            sent[idx] = sen
        cnt+=1
    
    for x in sent:
        sent[x] /= cnt
    fig = plt.figure(figsize=(20,5))
    st.header('average tweet sentiment of authors who have used the word ' +word)
    plt.xticks(rotation=90)
    plt.xlabel('author_username')
    plt.ylabel('average sentiment of tweets')
    sns.barplot(x=list(sent.keys()), y=list(sent.values()))
    st.pyplot(fig)

    if len(idxs) > 1:
        similar_tweets = {'pair of authors ids' : [] , 'tweet 1' : [], 'tweet 2' : [] , 'similarity value' : []}
        nlp = spacy.load("en_core_web_lg")
        ids = dataX_text.tweet_id.values
        corpus = dataX_text.text.values
        logging.set_verbosity(logging.ERROR)
        text_embeddings = fnc.get_embeding(corpus)
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                sim = cosine_similarity(np.array(text_embeddings[i]).reshape(1,-1), np.array(text_embeddings[j]).reshape(1,-1))
                if sim > 0.7 and dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0] != dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]:
                    author_idxs = []
                    author_idxs.append(dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0])
                    author_idxs.append(dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0])
                    author_idxs.sort()
                    for i in range(len(author_idxs)):
                        author_idxs[i] = authors[authors.id == author_idxs[i]]['username'].values[0]
                    author_idxs = tuple(author_idxs)
                    similar_tweets['pair of authors ids'].append(author_idxs)
                    similar_tweets['tweet 1'].append(dataX_text[dataX_text.tweet_id == ids[i]]['text'].values[0])
                    similar_tweets['tweet 2'].append(dataX_text[dataX_text.tweet_id == ids[j]]['text'].values[0])
                    similar_tweets['similarity value'].append(sim)

        if len(similar_tweets) > 1:
            similar_Tweets_df = pd.DataFrame.from_dict(similar_tweets)
            st.download_button(
                label="Download Similar Tweets as CSV",
                data=convert_df(similar_Tweets_df),
                file_name='authors_similarities.csv',
                mime='text/csv',
            )