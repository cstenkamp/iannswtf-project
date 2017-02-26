# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:11:51 2017

@author: csten_000
"""
import pickle
from pathlib import Path
import numpy as np
import collections

   
#===========================================

    
def random_strings(dataset, amount, primitive=False, printstuff=False):
    def myindex(inwhere, what):
        try:
            return inwhere.index(what)
        except ValueError:
            return len(inwhere)
    randomstrings = []
    if primitive:
        strings = list(dataset.uplook.keys())
        lengths = [myindex(curreview,dataset.ohnum) for curreview in dataset.trainreviews]
        for i in range(amount):
            length = lengths[np.random.randint(len(lengths))]
            randomstrings.append(np.random.choice(strings,length))
    else: #lass die wahrscheinlichkeitsverteilung der wörter realistisch sein
        allwords = []
        for curreview in dataset.trainreviews:
            allwords.extend(curreview)     
        count = []
        count.extend(collections.Counter(allwords).most_common(999999999))
        if printstuff: print("Most common words:", [dataset.uplook[i[0]] for i in count[:5]])
        counter = 0
        for i in range(len(count)): #jetzt sind die zähler die absolute summe. wenn wir jetzt ne zufallszahl zwischen 0 und abslen ziehen, haben wir korrekte verteilung.
            counter += count[i][1]
            count[i] = count[i][0], counter
        lengths = [myindex(curreview,dataset.ohnum) for curreview in dataset.trainreviews]
        for i in range(amount):
            length = lengths[np.random.randint(len(lengths))]
            currstring = []
            for j in range(length):
                num = np.random.randint(len(allwords))
                for k in range(len(count)):
                    if num < count[k][1]:
                        currstring.append(count[k][0])
                        break
            randomstrings.append(list(currstring))
    return randomstrings
        


if __name__ == '__main__':    
    checkpointpath = "./trumpdatweights/"
    assert Path(checkpointpath+"dataset_mit_wordvecs.pkl").is_file()
    print("Dataset including word2vec found!")
    with open(checkpointpath+'dataset_mit_wordvecs.pkl', 'rb') as input:
        datset = pickle.load(input)   
    strings = random_strings(datset, 5, False)
    print("The randomly generated tweets:")
    print([[datset.uplook[elem] for elem in currstring]for currstring in strings])