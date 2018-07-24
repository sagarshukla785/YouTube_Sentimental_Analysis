# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:52:33 2018

@author: Sagar Shukla
"""

import re
import sys
import time
import json
import requests
import lxml.html
import random
import pickle
import os.path
import nltk
from nltk.corpus import stopwords, movie_reviews 
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build
from lxml.cssselect import CSSSelector

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
DEVELOPER_KEY = "AIzaSyDhWZjOQ_lCKHbJJnE14fKx1Vor8dCBBFI"

# The function takes the video id and returns the no of likes and dislikes
# for that video.

def getLikesDislikes(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    results = youtube.videos().list(
        part = "statistics",
        id = video_id,
    ).execute()

    likes = int(results["items"][0]["statistics"]["likeCount"])
    dislikes = int(results["items"][0]["statistics"]["dislikeCount"])

    return likes,dislikes


# Here form this to line no. 183 each of the function has used to fetch the comments for the
# given video id.

YOUTUBE_COMMENTS_URL = 'https://www.youtube.com/all_comments?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_ajax'
USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'


def find_value(html, key, num_chars=2):
    pos_begin = html.find(key) + len(key) + num_chars
    pos_end = html.find('"', pos_begin)
    return html[pos_begin: pos_end]


def extract_comments(html):
    tree = lxml.html.fromstring(html)
    item_sel = CSSSelector('.comment-item')
    text_sel = CSSSelector('.comment-text-content')
    time_sel = CSSSelector('.time')
    author_sel = CSSSelector('.user-name')

    for item in item_sel(tree):
        id = item.get('data-cid')
        if '.' not in id:
            yield {'cid': item.get('data-cid'),
                   'text': text_sel(item)[0].text_content(),
                   'time': time_sel(item)[0].text_content().strip(),
                   'author': author_sel(item)[0].text_content()}


def extract_reply_cids(html):
    tree = lxml.html.fromstring(html)
    sel = CSSSelector('.comment-replies-header > .load-comments')
    return [i.get('data-cid') for i in sel(tree)]


def ajax_request(session, url, params, data, retries=10, sleep=20):
    for _ in range(retries):
        response = session.post(url, params=params, data=data)
        if response.status_code == 200:
            response_dict = json.loads(response.text)
            return response_dict.get('page_token', None), response_dict['html_content']
        else:
            time.sleep(sleep)


def download_comments(youtube_id, sleep=1):
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    # Get Youtube page with initial comments
    response = session.get(YOUTUBE_COMMENTS_URL.format(youtube_id=youtube_id))
    html = response.text
    reply_cids = extract_reply_cids(html)

    ret_cids = []
    for comment in extract_comments(html):
        ret_cids.append(comment['cid'])
        yield comment

    page_token = find_value(html, 'data-token')
    session_token = find_value(html, 'XSRF_TOKEN', 4)

    first_iteration = True

    # Get remaining comments (the same as pressing the 'Show more' button)
    while page_token:
        data = {'video_id': youtube_id,
                'session_token': session_token}

        params = {'action_load_comments': 1,
                  'order_by_time': False,
                  'filter': youtube_id}

        if first_iteration:
            params['order_menu'] = True
        else:
            data['page_token'] = page_token

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        page_token, html = response

        reply_cids += extract_reply_cids(html)
        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment

        first_iteration = False
        time.sleep(sleep)

    # Get replies (the same as pressing the 'View all X replies' link)
    for cid in reply_cids:
        data = {'comment_id': cid,
                'video_id': youtube_id,
                'can_reply': 1,
                'session_token': session_token}

        params = {'action_load_replies': 1,
                  'order_by_time': False,
                  'filter': youtube_id,
                  'tab': 'inbox'}

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        _, html = response

        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment
        time.sleep(sleep)


def getVideoComments(video_id):
    try:

        print('Downloading Youtube comments for video:', video_id)
        count = 0
        youtube_comments = []

        for comment in download_comments(video_id):
            youtube_comments.append(comment['text'])
            #print(json.dumps(comment['text']), file=fp)
            count += 1
            if count >= 200:
                break
            sys.stdout.write('Downloaded %d comment(s)\r' % count)
            sys.stdout.flush()
        #print("\nDone")

        return youtube_comments

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)
        
# This function takes the video url and extracts the video_id
# from it.
        
def getVideoId(video_url):
    pattern1  = "^https://www\.youtube\.com/watch\?v=.*$"
    pattern2 = "www\.youtube\.com/watch\?v=.*$"
   
    
    if re.match(pattern1, video_url) or re.match(pattern2, video_url):
        pos = video_url.find('=')
        id = video_url[pos+1:]
        return id
    
    else:
        return None
        
# This function performs some kind of pre-processing on the data i.e. comments in this case.
# It removes all the stop words and other unnecesarry words from the comments which do not
# perform the major role towards the analysis of the text.
        
def filterComments(comments):
    for i in range(0,len(comments)):
        comments[i] = comments[i].strip('\ufeff')
        comments[i] = comments[i].lower()


    stop_words = set(stopwords.words('english'))
    filtered_comments = []
    for comment in comments:
        word_tokens = word_tokenize(comment)
        links = [w for w in word_tokens if w.startswith('www.') or w.startswith('http')]
        mentions = [w for w in word_tokens if w.startswith('@')]

        filter = [w for w in word_tokens if w not in stop_words and w not in links and w not in mentions]
        filtered_comments.append(filter)

    return filtered_comments
    
    
word_features = []

# This function finds the features for the words from the dataset.

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
    
    
# This function uses the movie_review dataset in order to classify the model using
# naive bayes classifier.
    
def trainClassifier():
    if(os.path.isfile("naivebayes.pickle")):
        return;
        
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((list(movie_reviews.words(fileid)),category))
            
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    
    
    for w in all_words.most_common(10000):
        if(len(w[0]) >= 3):
            word_features.append(w[0])
            
                    
    feature_sets = [(find_features(rev), category) for (rev, category) in documents]
    
    random.shuffle(feature_sets)   
           
    classifier = nltk.NaiveBayesClassifier.train(feature_sets[:2000])
    
    save_classifier = open("naivebayes.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    
    
    
def bagOfWords(words):
    return dict([word, True] for word in words)
    
    
# This is the function where the final calculation happens.
# It takes 5 parameters that are no. of positive comments,
# no of negative comments, no of likes, no. of dislikes and comments ratio. And
# Finally returns the sentiment.
    
def finalScore(pos, neg, no_of_likes, no_of_dislikes, comments_ratio):
    comments_pos_per = None
    comments_neg_per = None

    if pos == 0 and neg == 0:
        comments_pos_per = 0
        comments_neg_per = 0
        
    elif pos == 0:
        comments_pos_per = 0
        comments_neg_per = 100
        
    elif neg == 0:
        comments_pos_per = 100
        comments_neg_per = 0
        
    else:
        comments_pos_per = float(pos*100/(pos+neg))
        comments_neg_per = 100 - comments_pos_per
        
    
    comments_likes_per = (no_of_likes*100/(no_of_likes + no_of_dislikes))
    comments_dislikes_per = 100 - comments_likes_per
    

    comments_pos_per = (comments_pos_per * comments_ratio) / 100
    comments_neg_per = (comments_neg_per * comments_ratio) / 100
    
    likes_dislikes_ratio = 1 - (comments_ratio / 100)
    
    comments_likes_per = comments_likes_per * likes_dislikes_ratio
    comments_dislikes_per = comments_dislikes_per * likes_dislikes_ratio
    
    pos_sentiment = comments_pos_per + comments_likes_per

   # neg_sentiment = comments_neg_per + comments_dislikes_per
    
    if pos_sentiment > 45 and pos_sentiment < 60:
        return "mixed"
    elif pos_sentiment >= 60:
        return "positive"
    else:
        return "negative"

    
# This is the function where the sentiment of the video is defined.
# First of all the function fetches top 200 rated comments and filter each 
# of them.
# Then it uses the NaiveBayes Classifier to train the model. For traing the model
# we used the movie_review data set present in nltk.
# Then it categorise whether a comment is potive or negtive and count them.
# Finally it fetches the total no. of likes and dislikes of the video using youtube api and
# gives the sentiment of the video using 5 parameters that are no. of positive comments,
# no of negative comments, no of likes, no. of dislikes and comments ratio.
    
def getSentiment(video_id, comments_ratio):
    
    pos = 0
    neg = 0
    
    video_comments = getVideoComments(video_id)
    
    filtered_comments = filterComments(video_comments)
    
    trainClassifier()
    
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    
    for comment in filtered_comments:
        result = classifier.classify(bagOfWords(comment))
        
        if result == "pos":
            pos += 1
            
        else:
            neg += 1
            
    
        
    no_of_likes , no_of_dislikes = getLikesDislikes(video_id)

    result = finalScore(pos, neg, no_of_likes, no_of_dislikes, comments_ratio)

    return result
    
    

# Driver code.
# This function takes the URL of the YouTube video and value of comments_ration
# and then finally prints the sentiment of the video.

if __name__ == "__main__":
    
    youtube_video_url = input("Enter the URL of the video :- ")
    
    comments_ratio = float(input("Enter % (out of 100) you want to give to sentiment from comments : "))
    
    if comments_ratio > 100 or comments_ratio <= 0:
        print("Commnets ratio should be grater than 0 and less than or equal to 100");
        exit(-1);
        
    video_id = getVideoId(youtube_video_url)
    
    if(video_id is None):
        print("Please enter the valid URL for the video");
    
    else:
        print("Sentiment towards the video is ==> ", getSentiment(video_id, comments_ratio))
    
    