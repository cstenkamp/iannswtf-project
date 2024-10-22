# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:20:00 2017

@author: csten_000
"""
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np
import os
import collections


class thedataset(object):
    def __init__(self, trainx, trainy, testx, testy, validx, validy, lookup, uplook, count):
        self.trainreviews = trainx
        self.traintargets = trainy
        self.testreviews = testx
        self.testtargets = testy
        self.validreviews = validx
        self.validtargets = validy
        self.lookup = lookup
        self.uplook = uplook
        self.ohnum = count+1  #len(lookup)
        
        
    def add_wordvectors(self, wordvecs):
        self.wordvecs = wordvecs


    def showstringlenghts(self, whichones, percentage, printstuff): #if percentage is 1, its the maxlen of the entire dataset
        lens = []
        if whichones[0]:
            for i in self.trainreviews:
                lens.append(len(i))
        if whichones[1]:
            for i in self.testreviews:
                lens.append(len(i))
        if whichones[2]:
            for i in self.validreviews:
                lens.append(len(i))
        bins = np.arange(0, 1001, 50)   #bins = np.arange(0, max(lens), 75)
        if printstuff: 
            plt.xlim([min(lens)-5, 1000+5])  #plt.xlim([min(lens)-5, max(lens)+5])
            plt.hist(lens, bins=bins, alpha=0.5)
            plt.title('Lenghts of the strings')
            plt.show()
        lens.sort()
        return lens[(round(len(lens)*percentage))-1]
    
    
    #TODO: vielleicht nicht mittem im satz brechen lassen?
    def shortendata(self, whichones, percentage, lohnenderstring, printstuff, embedding_size):
        maxlen = self.showstringlenghts(whichones,percentage,printstuff) #75% of data has a maxlength of 312, soo...
        if printstuff: 
            print("Shortening the Strings...")
            print("Max.length: ",self.showstringlenghts(whichones,1,False))             
            print("Amount: ", len(self.trainreviews) if whichones[0] else 0 + len(self.testreviews) if whichones[1] else 0  + len(self.validreviews)  if whichones[2] else 0)
            
        if whichones[0]:     
            i = 0
            while True:
                if len(self.trainreviews[i]) > maxlen:
                    if len(self.trainreviews[i][maxlen+1:]) > lohnenderstring:
                        self.trainreviews.append(self.trainreviews[i][maxlen+1:])
                        self.traintargets.append(self.traintargets[i])
                    self.trainreviews[i] = self.trainreviews[i][:maxlen]
                if len(self.trainreviews[i]) < lohnenderstring:
                    del(self.trainreviews[i])
                    del(self.traintargets[i])
                else: #wichtig! wenn er es löscht, hat das danach jetzt den bereits genutzten index -> don't increase!
                    i = i+1
                if i >= len(self.trainreviews):
                    break
        if whichones[1]:     
            i = 0
            while True:
                if len(self.testreviews[i]) > maxlen:
                    if len(self.testreviews[i][maxlen+1:]) > lohnenderstring:
                        self.testreviews.append(self.testreviews[i][maxlen+1:])
                        self.testtargets.append(self.testtargets[i])
                    self.testreviews[i] = self.testreviews[i][:maxlen]
                if len(self.testreviews[i]) < lohnenderstring:
                    del(self.testreviews[i])
                    del(self.testtargets[i])
                else: #wichtig! wenn er es löscht, hat das danach jetzt den bereits genutzten index -> don't increase!                    
                    i = i+1
                if i >= len(self.testreviews):
                    break  
        if whichones[2]:     
            i = 0
            while True:
                if len(self.validreviews[i]) > maxlen:
                    if len(self.validreviews[i][maxlen+1:]) > lohnenderstring:
                        self.validreviews.append(self.validreviews[i][maxlen+1:])
                        self.validtargets.append(self.validtargets[i])
                    self.validreviews[i] = self.validreviews[i][:maxlen]
                if len(self.validreviews[i]) < lohnenderstring:
                    del(self.validreviews[i])
                    del(self.validtargets[i])
                else: #wichtig! wenn er es löscht, hat das danach jetzt den bereits genutzten index -> don't increase!
                    i = i+1
                if i >= len(self.validreviews):
                    break
        if printstuff: 
            print("to.....")  
            print(self.showstringlenghts(whichones,percentage,True))
            print("Amount: ", len(self.trainreviews) if whichones[0] else 0 + len(self.testreviews) if whichones[1] else 0  + len(self.validreviews)  if whichones[2] else 0)
            print("to.....")
        
        if whichones[0]:    
            for i in range(len(self.trainreviews)):
                if len(self.trainreviews[i]) < maxlen:
                    diff = maxlen - len(self.trainreviews[i])
                    self.trainreviews[i].extend([self.ohnum]*diff)
        if whichones[1]:    
            for i in range(len(self.testreviews)):
                if len(self.testreviews[i]) < maxlen:
                    diff = maxlen - len(self.testreviews[i])
                    self.testreviews[i].extend([self.ohnum]*diff)
        if whichones[2]:    
            for i in range(len(self.validreviews)):
                if len(self.validreviews[i]) < maxlen:
                    diff = maxlen - len(self.validreviews[i])
                    self.validreviews[i].extend([self.ohnum]*diff)
        
        if printstuff: print(self.showstringlenghts(whichones,percentage,True))   
        try:
            self.lookup["<END>"]
        except KeyError:
            self.lookup["<END>"] = self.ohnum
            self.uplook[self.ohnum] = "<END>"
            self.ohnum += 1 #nur falls noch kein end-token drin ist+1!
        if hasattr(self, 'wordvecs'): #falls man es NACH word2vec ausführt
            self.wordvecs = np.append(self.wordvecs,np.transpose(np.transpose([[0]*embedding_size])),axis=0)
        self.maxlenstring = maxlen
        return maxlen
    

    def closeones(self, indices):
        for i in indices:
            top_k = 5  # number of nearest neighbors
            dists = np.zeros(self.wordvecs.shape[0])
            for j in range(len(self.wordvecs)):
                dists[j] = cosine(self.wordvecs[i],self.wordvecs[j])
            dists[i] = float('inf')
            clos = np.argsort(dists)[:top_k]
            return [self.uplook[i] for i in clos]
    
    
    def printcloseones(self, word):
        print("Close to '",word.replace(" ",""),"': ",self.closeones([self.lookup[word]]))
    
               
    def prepareback(self, str):
        str = str.replace(" <comma>", ",")
        str = str.replace(" <colon>", ":")
        str = str.replace(" <openBracket>", "(")
        str = str.replace(" <closeBracket>", ")")
        str = str.replace(" <dots>", "...")
        str = str.replace(" <dot>", ".")
        str = str.replace(" <semicolon>", ";")
        str = str.replace("<quote>", '"')
        str = str.replace(" <question>", "?")
        str = str.replace(" <exclamation>", "!")
        str = str.replace(" <hyphen> ","-")
        str = str.replace(" <END>", "")
        str = str.replace(" <SuperQuestion>", "???")
        str = str.replace(" <SuperExclamation>", "!!!")
        return str
        
        
    def showarating(self,number):
        array = [self.uplook[i] for i in self.trainreviews[number]]
        str = ' '.join(array)
        str = self.prepareback(str)
        return str
        


    def read_words_as_id(self, where): #pendant to _file_to_word_ids
        result = []
        for string in where:
             for word in string:
                result.append(word)
             result.append("1")
        return result
    
    
    def read_words(self, where): #pendant to _read_words
        result = []
        for string in where:
             for word in string:
                result.append(self.uplook[word])
             result.append("<EOS>")
        return result
    
    
    def return_all(self, only_positive=False): #pendant to ptb_raw_data
        if only_positive:
            tmptr = [self.trainreviews[i] for i in range(len(self.traintargets)) if self.traintargets[i]==1]
            trains = self.read_words_as_id(tmptr)
            tmpte = [self.testreviews[i] for i in range(len(self.testtargets)) if self.testtargets[i]==1]
            tests = self.read_words_as_id(tmpte)
            tmpva = [self.validreviews[i] for i in range(len(self.validtargets)) if self.validtargets[i]==1]
            valids = self.read_words_as_id(tmpva)
        else:
            trains = self.read_words_as_id(self.trainreviews)
            tests = self.read_words_as_id(self.testreviews)
            valids = self.read_words_as_id(self.validreviews)
            
        return trains, tests, valids, self.ohnum


    def build_vocab(self): #pendant to _build_vocab
        return self.lookup

    #TODO: es kann sein dass das hier in der selben reihenfolge sein muss wie die zahlen!
    def get_vocab(self): #pendant to get_vocab
        return list(self.lookup.keys())


    def grammar_iterator(self, raw_data, batch_size, num_steps):
      raw_data = np.array(raw_data, dtype=np.int32)
      data_len = len(raw_data)
      batch_len = data_len // batch_size
      data = np.zeros([batch_size, batch_len], dtype=np.int32)
      for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
      epoch_size = (batch_len - 1) // num_steps
      if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
      for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)






        
## first we create, save & load the words as indices.
def make_dataset(whichsets = [True, True, True], config=None):
    assert os.path.exists(config.setpath)
    allwords = {}
    wordcount = 2
    datasets = [config.TRAINNAME, config.TESTNAME, config.VALIDATIONNAME]
    
    #first we look how often each word occurs, to delete single occurences.
    for currset in range(3):
        if whichsets[currset]:
            with open(config.setpath+datasets[currset]+".txt", encoding="utf8") as infile:
                string = []
                for line in infile: 
                    words = line.split()
                    for word in words:
                        string.append(word)
   
    #now we delete single occurences.
    count = []
    count2 = []
    count.extend(collections.Counter(string).most_common(999999999))
    for elem in count:
        if elem[1] > 1:
            count2.append(elem[0])
        
    print("Most common words:")
    print(count[0:5])

    
    #now we make a dictionary, mapping words to their indices
    for currset in range(3):
        if whichsets[currset]:    
            with open(config.setpath+datasets[currset]+".txt", encoding="utf8") as infile:
                for line in infile:
                    words = line.split()
                    for word in words:
                        if not word in allwords:
                            if word in count2: #words that only occur once don't count.
                                allwords[word] = wordcount
                                wordcount = wordcount +1
    #print(allwords)            
    
    #the token for single occurences is "<UNK>", the one for end of sentence (if needed) is reserved to 1
    allwords["<UNK>"] = 0
    allwords["<EOS>"] = 1            
    reverse_dictionary = dict(zip(allwords.values(), allwords.keys()))
        
    #now we make every ratings-string to an array of the respective numbers.
    ratings = [[],[],[]]
    for currset in range(3):
        if whichsets[currset]:        
            with open(config.setpath+datasets[currset]+".txt", encoding="utf8") as infile:        
                for line in infile:
                    words = line.split()
                    currentrating = []
                    if len(words) > 1:
                        for word in words:
                            try:
                                currentrating.append(allwords[word])
                            except KeyError:
                                currentrating.append(allwords["<UNK>"])
                        ratings[currset].append(currentrating)   
            
            
    #also, we make an array of the ratings
    ratetargets = [[],[],[]]
    for currset in range(3):
        if whichsets[currset]:     
            with open(config.setpath+datasets[currset]+"-target.txt", encoding="utf8") as infile:
                for line in infile:
                    if config.is_for_trump:
                        ratetargets[currset].append(int(line))
                    else:
                        if int(line) < 5:
                            ratetargets[currset].append(0)
                        else:
                            ratetargets[currset].append(1)
            
    #we made a dataset! :)
    datset = thedataset(ratings[0], ratetargets[0],
                         ratings[1], ratetargets[1],
                         ratings[2], ratetargets[2],
                         allwords, reverse_dictionary, wordcount)
    
    return datset



        