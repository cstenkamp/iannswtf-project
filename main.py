# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:22:43 2017

@author: csten_000
"""


import pickle
from pathlib import Path
import random
import numpy as np
#np.set_printoptions(threshold=np.nan)
import datetime
import os
import copy
import sys
import tweepy

#====own files====
import datasetclass
import word2vec
from lstmclass import plot_test_and_train, test_one_sample, validate, train_and_test
from create_random import random_strings
from downloadAndPreprocess import create_folder, run_all
from create_dataset import create_from_johannes
import generatornetwork
from tweepy_credentials import consumer_key, consumer_secret, access_key, access_secret

#==============================================================================

is_for_trump = True

#==============================================================================
 
class Config_moviedat(object):
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
    longruntrials = 11
    batch_size = 32
    expressive_run = False
    checkpointpath = "./moviedatweights/"    
    fast_create_antiset = False
    allnetworkinitscale = 0.1 
    generatorhiddensize = 200
    max_gen_loss_to_perform = 300
    min_disc_acc_to_perform = 0.7
    
    def __init__(self):
        if not os.path.exists(self.checkpointpath+"classifier/"):
            os.makedirs(self.checkpointpath+"classifier/")         
        if not os.path.exists(self.checkpointpath+"recognizer/"):
            os.makedirs(self.checkpointpath+"recognizer/")         
        if not os.path.exists(self.checkpointpath+"languagemodel/"):
            os.makedirs(self.checkpointpath+"languagemodel/")         
    
    
class Config_trumpdat(object):
    is_for_trump = True
    TRAINNAME = "train"
    TESTNAME = "test"
    VALIDATIONNAME = "validation"
    setpath = "./trumpsets/"
    w2v_usesets = [True, True, True]
    use_w2v = True
    embedding_size = 128
    num_steps_w2v = 200001 
    maxlen_percentage = .90
    minlen_abs = 15
    TRAIN_STEPS = 12
    longruntrials = 20
    batch_size = 48
    expressive_run = False
    checkpointpath = "./trumpdatweights/"
    fast_create_antiset = False
    allnetworkinitscale = 0.1 #kleiner falls mehr iterationen
    generatorhiddensize = 200 #könnte auch >1000 sein
    max_gen_loss_to_perform = 300
    min_disc_acc_to_perform = 0.7
    
    def __init__(self):
        if not os.path.exists(self.checkpointpath+"classifier/"):
            os.makedirs(self.checkpointpath+"classifier/")         
        if not os.path.exists(self.checkpointpath+"recognizer/"):
            os.makedirs(self.checkpointpath+"recognizer/")         
        if not os.path.exists(self.checkpointpath+"languagemodel/"):
            os.makedirs(self.checkpointpath+"languagemodel/")             
    
      

#==============================================================================

def to_one_hot(y):
    y_one_hot = []
    for row in y:
        if row == 0:
            y_one_hot.append([1.0, 0.0])
        else:
            y_one_hot.append([0.0, 1.0])
    return np.array([np.array(row) for row in y_one_hot])


def get_cmdarguments():
    flag_onlyrun = flag_deleteall = flag_longversion = flag_showeverything = flag_shutup = False
    if len(sys.argv) > 1:
        if "-onlyrun" in sys.argv: 
            flag_onlyrun = True
        else:
            flag_onlyrun = input("Do you just want to generate a tweet?") in ('y','yes','Y','Yes','YES')
            
        if not flag_onlyrun:
            if "-deleteall" in sys.argv:
                flag_deleteall = True
            else:
                flag_deleteall = input("Do you want to start completely from scratch?") in ('y','yes','Y','Yes','YES')
            if "-longversion" in sys.argv:
                flag_longversion = True
            else:
                flag_longversion = input("Do you want to run the long version, which figures out the right amount of training etc automatically?") in ('y','yes','Y','Yes','YES')
            if "-showeverything" in sys.argv:
                flag_showeverything = True
            else:
                flag_showeverything = input("Do you want to run the expressive mode, generating lots of output-information?") in ('y','yes','Y','Yes','YES')
    
        if "-shutup" in sys.argv:
            flag_shutup =True
        
    return flag_onlyrun, flag_deleteall, flag_longversion, flag_showeverything, flag_shutup

#==============================================================================


def load_dataset(config, include_w2v, include_tsne):    
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
                
        if include_w2v: #taken from https://www.tensorflow.org/tutorials/word2vec/
            print("Starting word2vec...")
            
            word2vecresult, w2vsamplecount = word2vec.perform_word2vec(config, datset)
            datset.add_wordvectors(word2vecresult)

            with open(config.checkpointpath+'dataset_mit_wordvecs.pkl', 'wb') as output:
                pickle.dump(datset, output, pickle.HIGHEST_PROTOCOL)
            print("Saved word2vec-Results.")
            print("Word2vec ran through",w2vsamplecount,"different strings.")
            if config.is_for_trump:
                datset.printcloseones("4")  #bei twitterdaten sind WIE ERWARTET "4" und "for" nah!!!
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





def prepare_dataset(config, datset, onlywith = 0, printstuff = False):
    
    if printstuff: 
        previous_maxlen = datset.showstringlenghts([True, True, True], 1, False)
    
    now_maxlen = datset.shortendata([True, True, True], config.maxlen_percentage, config.minlen_abs, config.expressive_run, config.embedding_size)
    
    if printstuff:
        print("So far, there are",len(datset.trainreviews),"strings...")
        print("Shortening from max.",previous_maxlen,"words to", now_maxlen,"words (min",str(config.minlen_abs)+")")
        print("...afterwards, there are",len(datset.trainreviews),"strings.")
        
    X_train = np.asarray(datset.trainreviews)
    y_train = to_one_hot(np.asarray(datset.traintargets))
    if onlywith > 0:
        X_train = np.concatenate([X_train[:onlywith//2], X_train[-onlywith//2:]]) 
        y_train = np.concatenate([y_train[:onlywith//2], y_train[-onlywith//2:]])
    
    X_test = np.asarray(datset.testreviews)
    y_test = to_one_hot(np.asarray(datset.testtargets))
    X_validat = np.asarray(datset.validreviews)
    y_validat = to_one_hot(np.asarray(datset.validtargets))
    
    percentage = sum([item[0] for item in y_train])/len([item[0] for item in y_train])*100
    if printstuff: print(round(percentage),"% of training-data is positive")
    assert 20 < percentage < 80, "The training data is bad for ANNs"
    return X_train, y_train, X_test, y_test, X_validat, y_validat


#==============================================================================


def create_antiset(config, datset, primitive=False, showsample = False):
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


def load_and_select_dataset(config, include_tsne = False, is_recognizer = False):
    datset = load_dataset(config, include_w2v = config.use_w2v, include_tsne = include_tsne)
    if is_recognizer:
        datset.traintargets = [1]*len(datset.traintargets)
        datset.testtargets = [1]*len(datset.testtargets)
        datset.validtargets = [1]*len(datset.validtargets)
        mergedsets = merge_sets(datset, create_antiset(config,datset,primitive=config.fast_create_antiset))    
        datset = mergedsets
    return datset
    
        
#==============================================================================

def remove_zwischengespeichertes(config):
    for whichdir in [config.checkpointpath, os.path.join(config.checkpointpath, "classifier"), os.path.join(config.checkpointpath, "recognizer"), os.path.join(config.checkpointpath, "languagemodel")]:
        for filename in os.listdir(whichdir):
            if Path(whichdir+filename).is_file():
                os.remove(os.path.join(whichdir, filename))
  
    
def reset_trump_dataset():
    create_folder("Tweets")
    run_all() #from downloadandpreprocess
    create_from_johannes("./")
    os.remove("./Trumpliker.txt")
    os.remove("./Trumphater.txt")
    os.remove("./Filtered Tweets Positive.txt")
    os.remove("./Filtered Tweets negative.txt")



#=============================================================================

#Functions of LSTM-class:
#def plot_test_and_train(config, dataset, amount_iterations, X_train, y_train, X_test, y_test, is_recognizer = False):
#def test_one_sample(config, dataset, string, doprint=False, is_recognizer = False):
#def validate(config, dataset, X_validat, y_validat, bkpath = "", is_recognizer = False):
#def train_and_test(config, dataset, amount_iterations, X_test, y_test, X_train, y_train, is_recognizer = False):
   

#==============================================================================

def check_disc_accuracy(config, is_recognizer=False):
    global checked_rec_acc_already, checked_cla_acc_already
    
    if is_recognizer:
        if checked_rec_acc_already: 
            return True
    
        if perform_classifier(config, validate_only=True, is_recognizer=True) < config.min_disc_acc_to_perform:
            print("The recognizer is too bad, at first we have to make it learn!")
            perform_classifier(config, short_run=True, is_recognizer=True)
    
        checked_rec_acc_already = True
        return True
    
    else:
        if checked_cla_acc_already: 
            return True
    
        if perform_classifier(config, validate_only=True, is_recognizer=False) < config.min_disc_acc_to_perform:
            print("The classifier is too bad, at first we have to make it learn!")
            perform_classifier(config, short_run=True, is_recognizer=False)  
    
        checked_cla_acc_already = True
        return True


def check_gen_accuracy(config):
    global checked_gen_acc_already
    
    if checked_gen_acc_already:
        return True
        
    if perform_generator(config, validate_only=True) > config.max_gen_loss_to_perform:
        print("The LanguageModel is too bad yet, at first we have to make it learn!")
        perform_generator(config, short_run=True)
        
    checked_gen_acc_already = True
    return True
    
    
#==============================================================================
#==============================================================================
#==============================================================================    
#==============================================================================    


def perform_generator(config, validate_only=False, long_run=False, delete_all=False, short_run=False):
    print("Looking at the Languagemodel/Generator...")

    datset = load_dataset(config, config.use_w2v, False)

    if validate_only:
        return generatornetwork.validate(datset, config, printstuff=True)

    if delete_all:
        remove_zwischengespeichertes(config)
        if config.is_for_trump:
            reset_trump_dataset()
            
    if long_run:
        generatornetwork.run_till_loss_lowerthan(datset, config, generatornetwork.LearnConfig())
    elif short_run:
        generatornetwork.run_train_and_valid(datset, config, generatornetwork.LearnConfig())



def perform_generator_generate(config, harsh_rules=True, checkaccuracy=True):
    
    datset = load_dataset(config, config.use_w2v, False)
    
    if checkaccuracy:
          check_gen_accuracy(config)
          if harsh_rules:
              check_disc_accuracy(config, is_recognizer=True)
              check_disc_accuracy(config, is_recognizer=False)
          
    return get_something_to_tweet(config, datset, harsh_rules=harsh_rules)[0]


    
def perform_classifier(config, is_recognizer=False, validate_only=False, long_run=False, short_run=False, delete_all=False):
    subfolder = "recognizer/" if is_recognizer else "classifier/"
    
    if is_recognizer:
        print("Starting the actual LSTM... (performing the recognizer)")
    else:
        print("Starting the actual LSTM... (performing the classifier)")
              
    datset = load_and_select_dataset(config, include_tsne = False, is_recognizer = is_recognizer)
    X_train, y_train, X_test, y_test, X_validat, y_validat = prepare_dataset(config, datset)        
    
    if validate_only:
        return validate(config=config, dataset=datset, X_validat=X_validat, y_validat=y_validat, bkpath=config.checkpointpath+subfolder, is_recognizer=is_recognizer)
        
    if delete_all:
        remove_zwischengespeichertes(config)
        if config.is_for_trump:
            reset_trump_dataset()
    
    if long_run:
        print("Best training-set-result after",plot_test_and_train(config=config, dataset=datset, amount_iterations=config.longruntrials, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, is_recognizer=is_recognizer),"iterations")
        try:
            validate(config=config, dataset=datset, X_validat=X_validat, y_validat=y_validat, bkpath=config.checkpointpath+subfolder+"ManyIterations/", is_recognizer=is_recognizer)
        except:
            print("Can't run on the validation set because you didn't agree to copy!")
            
    elif short_run:
        train_and_test(config=config, dataset=datset, amount_iterations=config.TRAIN_STEPS, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, is_recognizer=is_recognizer)
        validate(config=config, dataset = datset, X_validat=X_validat, y_validat=y_validat, bkpath = config.checkpointpath+subfolder, is_recognizer=is_recognizer)




def perform_classifier_on_string(config, string, is_recognizer = False, doprint = False, checkaccuracy=True):
    if checkaccuracy:
      check_disc_accuracy(config, is_recognizer=is_recognizer)
      
    if doprint: print("Testing the classifier on '", string, "'")
    
    datset = load_and_select_dataset(config, include_tsne = False, is_recognizer = is_recognizer)
    X_train, y_train, X_test, y_test, X_validat, y_validat = prepare_dataset(config, datset)   
    result = test_one_sample(config, datset, string, is_recognizer=is_recognizer)
    if doprint: print("Positive example" if result else "Negative example")
    return result
    


def get_something_to_tweet(config, dataset, howmany=1, minlen=4, harsh_rules=True): #diese Funktion kann theoretisch endlos laufen, but who cares.
    returntweets = []
    while len(returntweets) < howmany:
        tweets = generatornetwork.main_generate(config, dataset, howmany*2, nounk = True, avglen = 25)
        for tweet in tweets:
            if len(tweet) < 139:
                if harsh_rules:
                    if perform_classifier_on_string(config, tweet, doprint=False, is_recognizer=True):
                        if perform_classifier_on_string(config, tweet, doprint=False, is_recognizer=False):
                            allstartwithat = True
                            for word in tweet.split():
                                if word[0] != "@": allstartwithat = False
                            if not allstartwithat:      
                                if len(tweet.split()) >= minlen:
                                      returntweets.append(tweet)
                                      if len(returntweets) == howmany:
                                          break
                else:
                    returntweets.append(tweet)
                    if len(returntweets) == howmany:
                        break                    
    return returntweets



#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


if __name__ == '__main__':
    global flag_onlyrun, flag_deleteall, flag_longversion, flag_showeverything, flag_shutup
    global checked_rec_acc_already, checked_cla_acc_already, checked_gen_acc_already
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    try:
        
        #flag_onlyrun, flag_deleteall, flag_longversion, flag_showeverything, flag_shutup = get_cmdarguments()
        flag_onlyrun = True
        flag_deleteall = flag_longversion = flag_showeverything = flag_shutup = False
    
        if is_for_trump:
            config = Config_trumpdat()    
        else:
            config = Config_moviedat()    
    
        checked_rec_acc_already = checked_cla_acc_already = checked_gen_acc_already = False
        
        print("Using the","Trump" if config.is_for_trump else "Movie","dataset")
        
    
    #    print("VALIDATING THE DISCRIMINATOR")
    #    perform_classifier(config, validate_only=True, is_recognizer=False)   
    #    
    #    print("PERFORMING THE DISCRIMINATOR ON SOMETHING")
    #    perform_classifier_on_string(config, "@realdonaldtrump #MAGA", doprint=True)
    #
    #    print("GOING FOR THE GENERATOR, YEEEAHHHHH")
    #    print(perform_generator_generate(config))
        
        totweet = perform_generator_generate(config)
        print(totweet)
    
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)
        api.update_status(totweet)
        
    
    
    
    #    if config.is_for_trump:
    #        perform_classifier_on_string(config, "@realdonaldtrump #MAGA", True, is_recognizer=False)
    #        perform_classifier_on_string(config, "Cars now cheap here!", True, is_recognizer=False)
    #    else:
    #        perform_classifier_on_string(config, "I hated this movie. It sucks. The movie is bad, Worst movie ever. Bad Actors, bad everything.", True, is_recognizer=False)
    #        perform_classifier_on_string(config, "I loved this movie. It is awesome. The movie is good, best movie ever. good Actors, good everything.", True, is_recognizer=False)

    except ImportError:
        print("Import Errors. Did you download Tensorflow and Tweepy?")


    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

