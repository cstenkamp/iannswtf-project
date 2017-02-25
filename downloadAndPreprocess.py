# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:36:42 2017

@author: JoJo
"""

import tweepy
import json
import re
import os


#Twitter API credentials
consumer_key = "ixvNy9UqSi4nZMXZC51OnJdRo"
consumer_secret =  "WdU6p2icCQvmQWNwnYokTBIgmElUer3Xs1p4eoty5jIGvkwZfT"
access_key = "2617326211-Q3OeGSi5sJPaq4mHEGXJd2sPow2epBmLTBbdC0z"
access_secret = "O6umMNseUdPHJW8pX6CLUgONwhNul2NuD8oNqLoi7HKqL"

currentdir = os.getcwd()
#=============================================================================
'''Here, all parameter necessary to successfully gather, process and manage Tweets are created. This section also runs all other functions'''

def run_all():
    poslist = read_to_string('accountlist.txt')
    for p in poslist:
        get_all_tweets(p)
    mergetxt(True,'accountlist')
    
    neglist = read_to_string('negativeaccountlist.txt')
    for n in neglist:
       get_all_tweets(n)
    mergetxt(False,'negativeaccountlist')
    print('Done!')

def merge_only():
    mergetxt(True,'accountlist')
    mergetxt(False,'negativeaccountlist')


def create_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        
def read_to_string(name):
    try:
        array = []
        with open(name, encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if len(line) > 0: array.append(line)
            return array
    except FileNotFoundError:
        return []

#=============================================================================
'''This section of the code implements a twitter crawler using tweepy: http://www.tweepy.org/'''

def get_all_tweets(screen_name):

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []

    
    print('Downloading Tweets from User: '+screen_name)
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

    
    #close the file
    print("Tweets saved to "+ screen_name +'.txt')
    file.close()

#=============================================================================
'''This section of the code deals with preprocessing and saving Tweets'''



def mergetxt(tmp, namelist):
    if tmp == True:
        alltxt = open('Trumpliker.txt', 'a')
    else:
        alltxt = open('Trumphater.txt','a')
        
    merginglist = read_to_string(namelist+'.txt')
    for f in merginglist:
        with open (f+'.txt') as infile:
            alltxt.write(infile.read())
    if tmp == True:
        filterforcontent(tmp,'Trumpliker')
    else:
        if tmp == False:
            filterforcontent(tmp,'Trumphater')
            
def filterforcontent(tmp, alltwts):
    try:
        if tmp == True:
            outfile = open('Filtered Tweets positive.txt','a')
        if tmp == False:
            outfile = open('Filtered Tweets negative.txt','a')
            
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
                        line = line.replace('#','')
                        line = line.replace('&amp','and')
                        line = line.replace('w/a','without')
                        line = line.replace('w/', 'with')
                        line = line[:-1]
                        if tmp == True:
                            trmpwds = ['Trump','POTUS','president','MAGA','@realDonaldTrump','CPAC']
                            if any(ext in line for ext in trmpwds):
                                outfile.write(line + '\n')
                        else: 
                            if tmp == False:
                                outfile.write(line + '\n')
    except FileNotFoundError:
        return []

#============================================================================
if __name__ == '__main__':
    #pass in the username of the account you want to download
    create_folder('Negative')
    create_folder('Positive')
    run_all()