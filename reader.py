# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:22:43 2017

@author: csten_000
"""
import pickle
from pathlib import Path

NAME = "test"



def make_batch():
    HOWMANY = 999999
    allwords = {}
    counter = 1
    wordcount = 0
    with open("sets/"+NAME+".txt", encoding="utf8") as infile:
        for line in infile:
            words = line.split()
            for word in words:
                if not word in allwords:
                    allwords[word] = wordcount
                    wordcount = wordcount +1
            counter = counter + 1
            if counter > HOWMANY: 
                break
        #print(allwords)
    with open("sets/"+NAME+".txt", encoding="utf8") as infile:        
        counter = 1
        ratings = []
        for line in infile:
            words = line.split()
            currentrating = []
            for word in words:
                currentrating.append(allwords[word])
            ratings.append(currentrating)
            counter = counter + 1
            if counter > HOWMANY: 
                break          
    #print(len(allwords))
    #return ratings
    with open("sets/"+NAME+"-target.txt", encoding="utf8") as infile:
        ratetargets = []
        for line in infile:
            if int(line) < 5:
                ratetargets.append(0)
            else:
                ratetargets.append(1)
    moviedat = moviedata(ratings,ratetargets,allwords)
    return moviedat


class moviedata(object):
    def __init__(self, reviews, targets, lookup):
        self.reviews = reviews
        self.targets = targets
        self.lookup = lookup
        self.ohnum = len(lookup)



my_file = Path("./ratings.pkl")
if not my_file.is_file():
    moviedat = make_batch()
    with open('ratings.pkl', 'wb') as output:
        pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
else:
    with open('ratings.pkl', 'rb') as input:
        moviedat = pickle.load(input)        
    
    
print(moviedat.reviews[4])
print(moviedat.targets[4])
print(moviedat.ohnum)