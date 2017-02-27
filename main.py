# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:22:43 2017

@author: csten_000
"""
#also re-doing this: https://arxiv.org/abs/1408.5882

import pickle
from pathlib import Path
import random
import numpy as np
#np.set_printoptions(threshold=np.nan)
import datetime
import os
import copy

#====own files====
import datasetclass
import word2vec
from lstmclass import plot_test_and_train, test_one_sample, validate, train_and_test
from create_random import random_strings
from downloadAndPreprocess import create_folder, run_all
from create_dataset import create_from_johannes

#==============================================================================

is_for_trump = True

#==============================================================================
 
class Config_datset(object):
    is_for_trump = False
    TRAINNAME = "train"
    TESTNAME = "test"
    VALIDATIONNAME = "validation"
    setpath = "./moviesets/"
    w2v_usesets = [True, True, True]
    use_w2v = False
    embedding_size = 128
    num_steps_w2v = 200001 #198000 ist einmal durchs ganze movieratings-dataset (falls nach word2vec gekürzt)
    maxlen_percentage = .75
    minlen_abs = 40
    TRAIN_STEPS = 6
    batch_size = 32
    expressive_run = False
    checkpointpath = "./moviedatweights/"    
    longruntrials = 11
    def __init__(self):
        if not os.path.exists(self.checkpointpath):
            os.makedirs(self.checkpointpath)         
    
    
class Config_trumpdat(object):
    is_for_trump = True
    TRAINNAME = "train"
    TESTNAME = "test"
    VALIDATIONNAME = "validation"
    setpath = "./trumpsets/"
    w2v_usesets = [True, True, True]
    use_w2v = True
    embedding_size = 128
    num_steps_w2v = 100001 
    maxlen_percentage = .90
    minlen_abs = 15
    TRAIN_STEPS = 18
    longruntrials = 16
    batch_size = 48
    expressive_run = False
    checkpointpath = "./trumpdatweights/"
    def __init__(self):
        if not os.path.exists(self.checkpointpath):
            os.makedirs(self.checkpointpath)            
    
    
if is_for_trump:
    config = Config_trumpdat()    
else:
    config = Config_datset()    
        

#==============================================================================

def to_one_hot(y):
    y_one_hot = []
    for row in y:
        if row == 0:
            y_one_hot.append([1.0, 0.0])
        else:
            y_one_hot.append([0.0, 1.0])
    return np.array([np.array(row) for row in y_one_hot])

#==============================================================================


def load_dataset(include_w2v, include_tsne):    
    print('Loading data...')
    
    if Path(config.checkpointpath+"dataset_mit_wordvecs.pkl").is_file():
        print("Dataset including word2vec found!")
        with open(config.checkpointpath+'dataset_mit_wordvecs.pkl', 'rb') as input:
            datset = pickle.load(input)       
    else:
        if Path(config.checkpointpath+"dataset_ohne_wordvecs.pkl").is_file():
            print("dataset without word2vec found.")
            with open(config.checkpointpath+'dataset_ohne_wordvecs.pkl', 'rb') as input:
                datset = pickle.load(input)  
            print(datset.ohnum," different words.")
        else:
            print("No dataset found! Creating new...")
            datset = datasetclass.make_dataset(config.w2v_usesets, config)
            #print("Shortening to", datset.shortendata([True, True, True], .75, 40, True, config.embedding_size))
            print(""+str(datset.ohnum)+" different words.")
            rand = round(random.uniform(0,len(datset.traintargets)))
            print('Sample string', datset.trainreviews[rand][0:100], [datset.uplook[i] for i in datset.trainreviews[rand][0:100]])
            
            with open(config.checkpointpath+'dataset_ohne_wordvecs.pkl', 'wb') as output:
                pickle.dump(datset, output, pickle.HIGHEST_PROTOCOL)
                print('Saved the dataset as Pickle-File')
                
        if include_w2v: #https://www.tensorflow.org/tutorials/word2vec/
            #TODO: CBOW statt skip-gram, da wir nen kleines dataset haben!
            print("Starting word2vec...")
            
            word2vecresult, w2vsamplecount = word2vec.perform_word2vec(config, datset)
            datset.add_wordvectors(word2vecresult)
            
            with open(config.checkpointpath+'dataset_mit_wordvecs.pkl', 'wb') as output:
                pickle.dump(datset, output, pickle.HIGHEST_PROTOCOL)
            print("Saved word2vec-Results.")
            print("Word2vec ran through",w2vsamplecount,"different strings.")
            if is_for_trump:
                datset.printcloseones("4")  #kann mir gut vorstellen dass bei twitterdaten "4" und "for" nah sind.
                datset.printcloseones("evil") #socialism, hach this dataset *_*
                datset.printcloseones("trump")
            else:
                datset.printcloseones("movie")
            datset.printcloseones("woman")
            datset.printcloseones("<dot>")
            datset.printcloseones("his")
            datset.printcloseones("bad")
            datset.printcloseones("three")

            if include_tsne: word2vec.plot_tsne(datset.wordvecs, datset, config.checkpointpath+'tsne.png')
            
    print('Data loaded.')
    return datset





def prepare_dataset(datset):
    print("So far, there are",len(datset.trainreviews),"strings...")
    print("Shortening from max.",datset.showstringlenghts([True, True, True], 1, False),"words to", datset.shortendata([True, True, True], config.maxlen_percentage, config.minlen_abs, config.expressive_run, config.embedding_size),"words (min",str(config.minlen_abs)+")")
    print("...afterwards, there are",len(datset.trainreviews),"strings.")
    #print("Max-Len:",datset.maxlenstring)
    X_train = np.asarray(datset.trainreviews)
    y_train = to_one_hot(np.asarray(datset.traintargets))
    #X_train = np.concatenate([X_train[:10000], X_train[12501:22501]])  #weg damit
    #y_train = np.concatenate([y_train[:10000], y_train[12501:22501]])
    
    X_test = np.asarray(datset.testreviews)
    y_test = to_one_hot(np.asarray(datset.testtargets))
    X_validat = np.asarray(datset.validreviews)
    y_validat = to_one_hot(np.asarray(datset.validtargets))
    
    percentage = sum([item[0] for item in y_train])/len([item[0] for item in y_train])*100
    print(round(percentage),"% of training-data is positive")
    assert 20 < percentage < 80, "The training data is bad for ANNs"
    return X_train, y_train, X_test, y_test, X_validat, y_validat





def run_lstm(datset, X_train, y_train, X_test, y_test, X_validat, y_validat):
    global previous_input_was
    print("Starting the actual LSTM...")
    
    if previous_input_was:
        print("Best training-set-result after",plot_test_and_train(config=config, dataset=datset, amount_iterations=config.longruntrials, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test),"iterations")
        try:
            validate(config=config, dataset=datset, X_validat=X_validat, y_validat=y_validat, bkpath=config.checkpointpath+"ManyIterations/")
        except:
            print("Can't run on the validation set because you didn't agree to copy!")
    else:
        train_and_test(config=config, dataset=datset, amount_iterations=config.TRAIN_STEPS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        validate(config=config, dataset = datset, X_validat=X_validat, y_validat=y_validat, bkpath = config.checkpointpath)
    
    if config.is_for_trump:
        test_one_sample(config, datset, "I hate immigrants", True)
        test_one_sample(config, datset, "I hate Trump", True)
    else:
        test_one_sample(config, datset, "I hated this movie. It sucks. The movie is bad, Worst movie ever. Bad Actors, bad everything.", True)
        test_one_sample(config, datset, "I loved this movie. It is awesome. The movie is good, best movie ever. good Actors, good everything.", True)





def create_antiset(datset, primitive=False, showsample = False):
    print("Creating an anti-dataset...")
    if Path(config.checkpointpath+"antiset_with_wordvecs.pkl").is_file():
        print("Antiset including word2vec found!")
        with open(config.checkpointpath+'antiset_with_wordvecs.pkl', 'rb') as input:
            antiset = pickle.load(input)       
            return antiset
            
    antitrain = random_strings(datset, len(datset.trainreviews), primitive)
    antitest = random_strings(datset, len(datset.testreviews), primitive)
    antivalid = random_strings(datset, len(datset.validreviews), primitive)
    
    antiset = datasetclass.thedataset(antitrain, [0]*len(datset.traintargets),
                                      antitest, [0]*len(datset.testtargets),
                                      antivalid, [0]*len(datset.validtargets),
                                      copy.deepcopy(datset.lookup), copy.deepcopy(datset.uplook), datset.ohnum)
    antiset.add_wordvectors(copy.deepcopy(datset.wordvecs))

    try:
        antiset.maxlenstring = datset.maxlenstring
    except:
        print("For both sets, the maxlenstring is missing!")
    
    with open(config.checkpointpath+'antiset_with_wordvecs.pkl', 'wb') as output:
        pickle.dump(antiset, output, pickle.HIGHEST_PROTOCOL)
        print('Saved the anti-dataset as Pickle-File')    
    
    if showsample:
        rand = round(random.uniform(0,len(datset.traintargets)))
        print('Sample string', antiset.trainreviews[rand][0:100], [antiset.uplook[i] for i in antiset.trainreviews[rand][0:100]])

    return antiset





def merge_sets(dataset, antiset):
    tr= copy.deepcopy(dataset.trainreviews); tr.extend(antiset.trainreviews)
    te= copy.deepcopy(dataset.testreviews); te.extend(antiset.testreviews)
    va= copy.deepcopy(dataset.validreviews); va.extend(antiset.validreviews)
    
    merged = datasetclass.thedataset(tr, [1]*len(dataset.traintargets)+[0]*len(antiset.traintargets),
                                     te, [1]*len(dataset.testreviews)+[0]*len(antiset.testreviews),
                                     va, [1]*len(dataset.validreviews)+[0]*len(antiset.validreviews),
                                     copy.deepcopy(dataset.lookup), copy.deepcopy(dataset.uplook), dataset.ohnum-1) #-1 cause it adds some itself
    merged.add_wordvectors(copy.deepcopy(dataset.wordvecs))    
    try:
        antiset.maxlenstring = dataset.maxlenstring if dataset.maxlenstring > antiset.maxlenstring else antiset.maxlenstring
    except:
        print("For both sets, the maxlenstring is missing!")    
    for i in range(len(merged.trainreviews)):
        merged.trainreviews[i] = list(merged.trainreviews[i]) 
    for i in range(len(merged.testreviews)):
        merged.testreviews[i] = list(merged.testreviews[i]) 
    for i in range(len(merged.validreviews)):
        merged.validreviews[i] = list(merged.validreviews[i])
    
    
    return merged


def perform_classifier():
    datset = load_dataset(config.use_w2v, False)
    X_train, y_train, X_test, y_test, X_validat, y_validat = prepare_dataset(datset)
    run_lstm(datset, X_train, y_train, X_test, y_test, X_validat, y_validat)

    
def perform_recognizer():
    datset = load_dataset(config.use_w2v, False)
    datset.traintargets = [1]*len(datset.traintargets)
    datset.testtargets = [1]*len(datset.testtargets)
    datset.validtargets = [1]*len(datset.validtargets)
    mergedsets = merge_sets(datset, create_antiset(datset,True))    
    X_train, y_train, X_test, y_test, X_validat, y_validat = prepare_dataset(mergedsets)        
    run_lstm(mergedsets, X_train, y_train, X_test, y_test, X_validat, y_validat)
        


def remove_zwischengespeichertes():
    for filename in os.listdir(config.checkpointpath):
        if Path(config.checkpointpath+filename).is_file():
            os.remove(os.path.join(config.checkpointpath, filename))



#=============================================================================

#Functions of LSTM-class:
#def plot_test_and_train(config, dataset, amount_iterations, X_train, y_train, X_test, y_test):
#def test_one_sample(config, dataset, string, doprint=False):
#def validate(config, dataset, X_validat, y_validat, bkpath = ""):
#def train_and_test(config, dataset, amount_iterations, X_test, y_test, X_train, y_train):
    
 
#==============================================================================


if __name__ == '__main__':
    global previous_input_was
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    print("Using the","Trump" if is_for_trump  else "Movie","dataset")
    
    previous_input_was = input("Do you want to run the long version, which figures out the right amount of training etc automatically?") in ('y','yes','Y','Yes','YES')
    
    if input("Do you want to start completely from scratch?") in ('y','yes','Y','Yes','YES'):
        remove_zwischengespeichertes()
        if is_for_trump:
            create_folder("Tweets")
            run_all() #from downloadandpreprocess
            create_from_johannes("./")
            os.remove("./Trumpliker.txt")
            os.remove("./Trumphater.txt")
            os.remove("./Filtered Tweets Positive.txt")
            os.remove("./Filtered Tweets Negative.txt")
            
    
    perform_classifier()    
    
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))



#==============================================================================
#OK, now the Generative Model. Yay
#https://arxiv.org/pdf/1609.05473.pdf
#https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/seq-gan.md






