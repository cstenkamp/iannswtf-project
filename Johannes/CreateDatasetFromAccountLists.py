# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:13:46 2017

@author: Johannes
"""

import tweepy
import json
import re


#Twitter API credentials
consumer_key = "ixvNy9UqSi4nZMXZC51OnJdRo"
consumer_secret =  "WdU6p2icCQvmQWNwnYokTBIgmElUer3Xs1p4eoty5jIGvkwZfT"
access_key = "2617326211-Q3OeGSi5sJPaq4mHEGXJd2sPow2epBmLTBbdC0z"
access_secret = "O6umMNseUdPHJW8pX6CLUgONwhNul2NuD8oNqLoi7HKqL"


def get_all_tweets(screen_name):
    """This function collects tweets from Twitter-users in 200 Tweet batches and saves them to a .txt file with the corresponding account name"""
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    alltweets = []    
    print('collecting tweets from user: '+screen_name)
    
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
                      
    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print("...%s tweets downloaded so far" % (len(alltweets)))
       
    file = open(screen_name + '.txt', "w", encoding="utf8")
    for status in alltweets:
        json.dump(status.text,file,sort_keys = True,indent = 4)
        file.write("\n")

def read_to_string(name):
    """This function parses a .txt file into an array, creating new array elements for each column in the .txt file"""
    try:
        array = []
        with open(name, encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if len(line) > 0: array.append(line)
            return array
    except FileNotFoundError:
        return []


def mergetxt(namelist, savefile):
    """This function merges the content of.txt files together in one .txt file"""
    alltxt = open(savefile +'.txt', 'a')
    merginglist = read_to_string(namelist+'.txt')
    for f in merginglist:
        with open (f+'.txt') as infile:
            alltxt.write(infile.read())
    filterforcontent(savefile)
            
def filterforcontent(alltwts):
    """This function prefilters the content of a .txt file. Because tweepy objects often copy parts of the Tweet that are non-text content, this content needs to be removed"""
    try:
        outfile = open(alltwts + '_filtered.txt','a')
        with open(alltwts+'.txt', encoding="utf8") as fltfile:
            for line in fltfile:
                if not 'http' in line:
                    if len(line) > 50:
                        while line.find("  ") > 0:
                            line = line.replace("  ","")
                        line = line[1:-1]
                        line = re.sub('(\\\\u\\w+?)+?\\b','',line)
                        line = line.replace('\\','')
                        line = line.replace('RT ','')
                        line = line.replace('&amp','and')
                        line = line.replace('w/o','without')
                        line = line.replace('w/', 'with')
                        line = line[:-1]
                        outfile.write(line + '\n')
    except FileNotFoundError:
        return []
    
if __name__ == '__main__':
    poslist = read_to_string('accountlist.txt')
    for p in poslist:
        get_all_tweets(p)
    mergetxt('accountlist','Tweets_positive')
    
    print('Finished collecting and processing positive Tweets!')
   
    neglist = read_to_string('negativeaccountlist.txt')
    for n in neglist:
       get_all_tweets(n)
    mergetxt('negativeaccountlist','Tweets_negative')
    print('Done')

   