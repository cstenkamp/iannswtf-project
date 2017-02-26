# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:36:44 2017

@author: Johannes
"""

"""This program passes the ANN output and creates a tweet with it!"""
import tweepy
import time

consumer_key = "ixvNy9UqSi4nZMXZC51OnJdRo"
consumer_secret =  "WdU6p2icCQvmQWNwnYokTBIgmElUer3Xs1p4eoty5jIGvkwZfT"
access_key = "2617326211-Q3OeGSi5sJPaq4mHEGXJd2sPow2epBmLTBbdC0z"
access_secret = "O6umMNseUdPHJW8pX6CLUgONwhNul2NuD8oNqLoi7HKqL"

def make_a_tweet():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    anntxt = open('filename.txt','r')
    ann = anntxt.readline(anntxt)
    for newstatus in ann:
        api.update_status(newstatus)
        time.sleep(10800)