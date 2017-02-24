# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:11:51 2017

@author: csten_000
"""
import pickle
from pathlib import Path
import numpy as np

#checkpointpath = "./trumpdatweights/"
#
#if Path(checkpointpath+"dataset_mit_wordvecs.pkl").is_file():
#    print("Dataset including word2vec found!")
#    with open(checkpointpath+'dataset_mit_wordvecs.pkl', 'rb') as input:
#        datset = pickle.load(input)   
#    
#    
#    strings = datset.uplook.keys()
#    lengths = [len(curreview) for curreview in dataset.trainreviews]
#    



def myindex(inwhere, what):
    try:
        return inwhere.index(what)
    except ValueError:
        return len(inwhere)

strings = [23,45,64,34]
trainreviews = [[3545,4536,346,12,45,76767,45,99,99],[12,45,64,3,675,324,99,99,99],[12,45,64,3,675,324,3,3,2]]
ohnum = 99

if True:
    
    lengths = [myindex(curreview,ohnum) for curreview in trainreviews]
    
    
    length = lengths[np.random.randint(len(lengths))]
    
    print(np.random.choice(strings,length))
    
    