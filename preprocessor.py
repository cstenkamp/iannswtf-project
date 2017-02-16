# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:28:19 2017

@author: Johannes
"""

import numpy as np
#import tensorflow as tf
import copy

class Tweet:
   'Here, Tweets are defined as objects for simpler handling'
   def __init__ (self,twhash,twid,userid,twcon):
        self.twhash = []
        self.twid = 0
        self.userid = 0
        self.twcon = []

def warmup():
    global fltlist
    words = open('wordfilter.txt')
    fltlist = words.readlines()

def filtertweets(twt):
    hlphst = []
    hlpcontent = []
    hst = twt.twhash
    content = twt.twcon
    
    for indo in hst:
        for indt in fltlist:
            if indo is not indt:
                hlphst.append(hst[indo])
    for oindo in content:
        for oindt in fltlist:
            if oindo is not oindt:
                hlpcontent.append(content[oindo])
                
    twt.twhash = hlphst
    twt.twcon = hlpcontent
                
    
fltlist = []
warmup()