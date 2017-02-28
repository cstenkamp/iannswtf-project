# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:36:42 2017

@author: JoJo
"""

import tweepy
import json
import re
import os
from pathlib import Path


#Create OAuth authentification to read tweets
from tweepy_credentials import consumer_key, consumer_secret, access_key, access_secret

#Folder location to read in twitter account lists
acclist = './accountlists/accountlist.txt'
deacclist = './accountlists/negativeaccountlist.txt'


#=============================================================================
'''Here, all parameter necessary to successfully gather, process and manage Tweets are created. This section also runs all other functions'''

def run_all():
    #to avoid tweets being preprocssed and passed into the final list multiple times, the raw .txt files are ersed in the beginning of each new session.
    if Path("./Trumpliker.txt").is_file():
        os.remove("./Trumpliker.txt")
    if Path("./Trumphater.txt").is_file():
        os.remove("./Trumphater.txt")
    
    #collecting tweets, merging and then preprocessing them happens once for the list of trump-supporters...
    poslist = read_to_string(acclist)
    for p in poslist:
        get_all_tweets(p)
    mergetxt(True, acclist)
    
    #...and for unrelated tweets
    neglist = read_to_string(deacclist)
    for n in neglist:
       get_all_tweets(n)
    mergetxt(False, deacclist)
    
    print('Done!')

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
'''This section of the code implements a twitter crawler using tweepy: http://www.tweepy.org/
Sections of this code are taken from this tutorial: https://nocodewebscraping.com/twitter-json-examples/
more specifically from this code: https://drive.google.com/file/d/0Bw1LIIbSl0xuNnJ0N1ppSkRjQjQ/view
Parallelizing the crawler is unfortunately not possible, since it would require a seperate OAuth-Key for every thread.
'''

def get_all_tweets(screen_name):

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []
    
    #if tweets of a specific account have already been downloaded, they are not downloaded again
    if os.path.isfile('./Tweets/' + screen_name+'.txt'):
        print('Tweets from '+screen_name+' already collected, moving on...')
        return
    
    print('Downloading Tweets from User: '+screen_name)
    
    #200 tweets are collected and then appended into a list
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    
    #download tweets as long as there are still tweets to download or until the max amount of tweets has been downloaded
    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print("...%s tweets downloaded so far" % (len(alltweets)))
        
        #when all tweets are downloaded, pass them into a .txt file with the according account name
        file = open('./Tweets/' + screen_name + '.txt', "w", encoding="utf8")
        for status in alltweets:
            json.dump(status.text,file,sort_keys = True,indent = 4)
            file.write("\n")
        
    print("Tweets saved to /Tweets/"+ screen_name +'.txt'+'\n')
    file.close()

#=============================================================================
'''This section of the code deals with preprocessing and saving Tweets into one file'''


'''the 'tmp' variable denotes whether the function is used to process data from the list of tweets of trump-supporter
or unrelated tweets. This holds for later functions as well and prevents unnecessary lines of code'''

def mergetxt(tmp, namelist, folder="./Tweets/"):
    if tmp == True:
        alltxt = 'Trumpliker.txt'
    else:
        alltxt = 'Trumphater.txt'
    
    #depending whether tmp is true or false, the raw content of an according .txt file is used for processing
    alltxtfile = open(alltxt,"a")
    merginglist = read_to_string(namelist)
    for f in merginglist:
            with open (folder+f+'.txt') as infile:
                alltxtfile.write(infile.read())
    filterforcontent(tmp, alltxt, "")
    
                                
def filterforcontent(tmp, alltwts, folder="./Tweets/"):
    #again checking for which tweets are being preprocessed to generate a clean tweet list
    try:
        if tmp == True:
            outfile = open('Filtered Tweets positive.txt','a')
        else: 
            if tmp == False:
                outfile = open('Filtered Tweets negative.txt','a')
            
        with open(folder+alltwts, encoding="utf8") as fltfile:
            for line in fltfile:
                if not 'http' in line:
                    if len(line) > 50:
                        while line.find("  ") > 0:
                            line = line.replace("  ","")
                        line = line[1:-1]
                        #This is necessary because all Tweets begin and end with "
                        line = re.sub('(\\\\u\\w+?)+?\\b','',line)
                        line = line.replace('\\','')
                        line = line.replace('RT ','')
                        #RT denotes "retweet". Even though retweets are not content created by the owner of the account, they still represent  political views and opinions
                        line = line.replace('&amp','and')
                        line = line.replace('w/a','without')
                        line = line.replace('w/', 'with')
                        line = line[:-1]
                        outfile.write(line + '\n')
    except FileNotFoundError:
        return []
#============================================================================
'''This function lets the user check how a string looks like after preprocessing. 
Note that preprocessing is handcrafted for tweepy string output and therefore may return odd-looking strings if the input a string that is a normal sentence
This function is never called when executing the code under normal circumstances but it provides an easy tool for users to process individual string manually'''

def processstring(line):
    if not 'http' in line:
        if len(line) > 50:
            while line.find("  ") > 0:
                line = line.replace("  ","")
            line = line[1:-1]
            line = re.sub('(\\\\u\\w+?)+?\\b','',line)
            line = line.replace('\\','')
            line = line.replace('RT ','')
            line = line.replace('&amp','and')
            line = line.replace('w/a','without')
            line = line.replace('w/', 'with')
            line = line[:-1]
            return line
        else:
            print('String too short!')
            return
    else:
        print('string has http in it and therefore will not be processed')
        return

#============================================================================
if __name__ == '__main__':
    create_folder('Tweets')
    run_all()