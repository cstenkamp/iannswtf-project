# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:22:43 2017

@author: csten_000
"""
#also re-doing this: https://arxiv.org/abs/1408.5882

import pickle
from pathlib import Path
import tensorflow as tf
import random
import numpy as np
#np.set_printoptions(threshold=np.nan)
import datetime
import matplotlib.pyplot as plt
import time
from scipy import stats
import os
import shutil

#====own functions====
import file_functions
from create_dataset import preparestring
import datasetclass
import word2vec


is_for_trump = True

 
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
    maxlen_percentage = 1
    minlen_abs = 10
    TRAIN_STEPS = 13
    longruntrials = 15
    batch_size = 48
    expressive_run = False
    checkpointpath = "./trumpdatweights/"
    def __init__(self):
        if not os.path.exists(self.checkpointpath):
            os.makedirs(self.checkpointpath)            
    
    
if is_for_trump:
    config = Config_trumpdat()    
else:
    config = Config_moviedat()    
        

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


print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))


def load_moviedata(include_w2v, include_tsne):    
    print('Loading data...')
    
    if Path(config.checkpointpath+"dataset_mit_wordvecs.pkl").is_file():
        print("Dataset including word2vec found!")
        with open(config.checkpointpath+'dataset_mit_wordvecs.pkl', 'rb') as input:
            moviedat = pickle.load(input)       
    else:
        if Path(config.checkpointpath+"dataset_ohne_wordvecs.pkl").is_file():
            print("dataset without word2vec found.")
            with open(config.checkpointpath+'dataset_ohne_wordvecs.pkl', 'rb') as input:
                moviedat = pickle.load(input)  
            print(moviedat.ohnum," different words.")
        else:
            print("No dataset found! Creating new...")
            moviedat = datasetclass.make_dataset(config.w2v_usesets, config)
            #print("Shortening to", moviedat.shortendata([True, True, True], .75, 40, True, config.embedding_size))
            print(""+str(moviedat.ohnum)+" different words.")
            rand = round(random.uniform(0,len(moviedat.traintargets)))
            print('Sample review', moviedat.trainreviews[rand][0:100], [moviedat.uplook[i] for i in moviedat.trainreviews[rand][0:100]])
            
            with open(config.checkpointpath+'dataset_ohne_wordvecs.pkl', 'wb') as output:
                pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
                print('Saved the dataset as Pickle-File')
                
        if include_w2v: #https://www.tensorflow.org/tutorials/word2vec/
            #TODO: CBOW statt skip-gram, da wir nen kleines dataset haben!
            print("Starting word2vec...")
            
            word2vecresult, w2vsamplecount = word2vec.perform_word2vec(config, moviedat)
            moviedat.add_wordvectors(word2vecresult)
            
            with open(config.checkpointpath+'dataset_mit_wordvecs.pkl', 'wb') as output:
                pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
            print("Saved word2vec-Results.")
            print("Word2vec ran through",w2vsamplecount,"different strings.")
            if is_for_trump:
                moviedat.printcloseones("4")  #kann mir gut vorstellen dass bei twitterdaten "4" und "for" nah sind.
                moviedat.printcloseones("evil") #socialism, hach this dataset *_*
                moviedat.printcloseones("trump")
            else:
                moviedat.printcloseones("movie")
            moviedat.printcloseones("woman")
            moviedat.printcloseones("<dot>")
            moviedat.printcloseones("his")
            moviedat.printcloseones("bad")
            moviedat.printcloseones("three")

            if include_tsne: word2vec.plot_tsne(moviedat.wordvecs, moviedat, config.checkpointpath+'tsne.png')
            
    print('Data loaded.')
    return moviedat


moviedat = load_moviedata(config.use_w2v, False)

print("So far, there are",len(moviedat.trainreviews),"strings...")
print("Shortening from max.",moviedat.showstringlenghts([True, True, True], 1, False),"words to", moviedat.shortendata([True, True, True], config.maxlen_percentage, config.minlen_abs, config.expressive_run, config.embedding_size),"words (min",str(config.minlen_abs)+")")
print("...afterwards, there are",len(moviedat.trainreviews),"strings.")
#print("Max-Len:",moviedat.maxlenstring)
X_train = np.asarray(moviedat.trainreviews)
y_train = to_one_hot(np.asarray(moviedat.traintargets))
#X_train = np.concatenate([X_train[:10000], X_train[12501:22501]])  #weg damit
#y_train = np.concatenate([y_train[:10000], y_train[12501:22501]])

X_test = np.asarray(moviedat.testreviews)
y_test = to_one_hot(np.asarray(moviedat.testtargets))
X_validat = np.asarray(moviedat.validreviews)
y_validat = to_one_hot(np.asarray(moviedat.validtargets))

percentage = sum([item[0] for item in y_train])/len([item[0] for item in y_train])*100
print(round(percentage),"% of training-data is positive")
assert 20 < percentage < 80

print("Starting the actual LSTM...")

                 

#=============================================================================
#OK. Now lets get to the actual LSTM, but using our pre-trained wordvectors.
#http://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow?rq=1


def create_batches(data_X, data_Y, batch_size):
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx : batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx : batch_size * (idx + 1)]
        yield x_batch, y_batch


class LSTM(object):
    def __init__(self, dataset, is_training):
    
        self.dataset = dataset
        self.input_data = tf.placeholder(tf.int32, [config.batch_size, self.dataset.maxlenstring], name="input_x")
        self.target = tf.placeholder(tf.float32, [config.batch_size, 2], name="input_t") #2 = n_classes
    
        #non-stateful LSTM   #128 ist hidden_size (=#Vectors???)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=0.0, state_is_tuple=True)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
            
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
        initial_state = cell.zero_state(config.batch_size, tf.float32)
            
        if config.use_w2v:
#            embedding = tf.Variable(self.dataset.wordvecs, name="embedding", dtype=tf.float32)
#            inputs = tf.nn.embedding_lookup(embedding, self.input_data)     
#            W = tf.Variable(tf.constant(0.0, shape=[self.dataset.ohnum, config.embedding_size]), trainable=False, name="W")
#            embedding_placeholder = tf.placeholder(tf.float32, [self.dataset.ohnum, config.embedding_size])
#            embedding_init = W.assign(embedding_placeholder)
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                embedding = tf.Variable(tf.random_uniform([self.dataset.ohnum, config.embedding_size], -1.0, 1.0), trainable = False, name="embedding")
                self.embedding = embedding
                inputs = tf.nn.embedding_lookup(embedding, self.input_data, name="embeddings")
        else:            
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self.dataset.ohnum+1, 128], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, self.input_data, name="embeddings")

        if is_training:
            inputs = tf.nn.dropout(inputs, 0.5)
            
        output, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        
        softmax_w = tf.get_variable("softmax_w", [128, 2], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.float32)
        logits = tf.matmul(last, softmax_w) + softmax_b
        self.logits = logits
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.target))
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.accuracy = accuracy
        
        if is_training:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
            self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        
            
            
    def run_on(self, session, x_data, y_data, is_training, saver=None, iteration=0, epoch=0, maxepoch=0, SaveALot=False):   
        step = 0
        acc_accuracy = 0
        for x_batch, y_batch in create_batches(x_data, y_data, config.batch_size):
            
            if is_training:
                print("Iteration: %d/%d; Progress: %d%%" % ((epoch+1),maxepoch,(round(step/(X_train.shape[0] // config.batch_size)*100))), end='\r')
                accuracy2, cost2, _ = session.run([self.accuracy, self.cost, self.train_op], feed_dict={self.input_data: x_batch, self.target: y_batch})
            else:
                print("Test-Run Progress: %d%%" % ((round(step/(X_test.shape[0] // config.batch_size)*100))), end='\r')
                accuracy2, cost2 = session.run([self.accuracy, self.cost], feed_dict={self.input_data: x_batch, self.target: y_batch})
            
            step += 1
            acc_accuracy += accuracy2
            
        accuracy = acc_accuracy / step
        
        if is_training:
            if config.use_w2v:
                savename = config.checkpointpath+"weights_wordvecs.ckpt"
            else:
                savename = config.checkpointpath+"weights.ckpt"
        
        if SaveALot:
            savename = config.checkpointpath+"ManyIterations/"
            if not os.path.exists(savename):
                os.makedirs(savename) 
            middlename = "_wordvecs" if config.use_w2v else ""
            savename += "weights"+middlename+"_iteration"+str(iteration+epoch+1)+".ckpt"
        
            saver.save(session, savename)
            
            time.sleep(0.1)
            file_functions.write_iteration(number = iteration+epoch+1, path=config.checkpointpath)
        
        return accuracy
    

def initialize_uninitialized_vars(session):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError: #rather ask for forgiveness than for allowance;
            uninitialized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)
            

def train_and_test(dataset, amount_iterations):
    file_functions.prepare_checkpoint(config.use_w2v,config.checkpointpath)
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
     #TODO: abfragen ob er lernen, applien oder beides will (oder vie-zeit-modus)
     #viel-zeit-modus: wo er train und test accuracy live errechnet und direkt plottet und man sich das beste aussuchen kann
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTM(dataset=dataset, is_training=True)
    
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
           
            ckpt = tf.train.get_checkpoint_state(config.checkpointpath) 
            if ckpt and ckpt.model_checkpoint_path:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
                iteration = file_functions.read_iteration(path = config.checkpointpath)
                print(iteration,"iterations ran already.")
            else:
                print("Created model with fresh parameters.")
                init = tf.global_variables_initializer()
                init.run()
                iteration = 0
            
            if config.use_w2v:
                session.run(model.embedding.assign(dataset.wordvecs))
                print("Using the pre-trained word2vec")
            else:
                print("Not using the pre-trained word2vec")
            

            training_steps = amount_iterations
            try:
                if iteration > 0:
                    if amount_iterations > iteration:
                        if not input(str(iteration)+" iterations ran already, "+str(amount_iterations)+" are supposed to run. Do you want an additional "+str(amount_iterations)+"(y), or just the "+str(amount_iterations-iteration)+" to fill up to "+str(amount_iterations)+" (n)? ") in ('y','yes','Y','Yes','YES'):
                            training_steps = amount_iterations - iteration
                    else:
                        if not input("It seems like all the requested "+str(amount_iterations)+" ran already. Do you want another "+str(amount_iterations)+" iterations to run? ") in ('y','yes','Y','Yes','YES'):
                            training_steps = 0
            except TypeError:
                training_steps = amount_iterations - iteration
                print("Training for another",training_steps,"iterations.")
                
            for i in range(training_steps):
                
                train_accuracy = model.run_on(session, X_train, y_train, True, saver, iteration, i, training_steps)
                print("Epoch: %d \t Train Accuracy: %.3f" % (i + 1, train_accuracy))          
        
            
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            print("Trying to Apply the model to the test-set...")        
            testmodel = LSTM(dataset=dataset, is_training=False)
            
            if config.use_w2v:
                session.run(testmodel.embedding.assign(dataset.wordvecs))
                print("Using the pre-trained word2vec")
            else:
                print("Not using the pre-trained word2vec")
                
                
            test_accuracy = testmodel.run_on(session, X_test, y_test, False)
            print("Testing Set Accuracy: %.3f" % (test_accuracy))          






def validate(dataset, bkpath = config.checkpointpath):
   with tf.Graph().as_default(), tf.Session() as session:
       initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
       with tf.variable_scope("model", reuse=None, initializer=initializer):
            print("Trying to apply the model to the validation-set...")        
            testmodel = LSTM(dataset=dataset, is_training=False)

            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
           
            ckpt = tf.train.get_checkpoint_state(bkpath) #TODO: da unterscheidet er noch nicht zwischen mit und ohne w2v..
            if ckpt and ckpt.model_checkpoint_path:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
                print(file_functions.read_iteration(path = bkpath),"iterations ran already.")
            else:
                print("uhm, without a model it doesn't work") #TODO: anders
                exit()
            
            if config.use_w2v:
                session.run(testmodel.embedding.assign(dataset.wordvecs))
                print("Using the pre-trained word2vec")
            else:
                print("Not using the pre-trained word2vec")
                
                
            valid_accuracy = testmodel.run_on(session, X_validat, y_validat, False)
            print("Validation Set Accuracy: %.3f" % (valid_accuracy))     
            
            
            
            
            

def test_one_sample(dataset, string, doprint=False):
    if doprint: print("Possible Text:",string)
    dataset = [dataset.lookup[i] if i in dataset.lookup.keys() else dataset.lookup["<UNK>"] for i in preparestring(string).split(" ")] #oh damn, für solche einzeiler liebe ich python.
    dataset = dataset + [0]*(dataset.maxlenstring-len(dataset))
    dataset = [dataset]*config.batch_size 
    data_t = to_one_hot([0]*config.batch_size)
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
        with tf.variable_scope("model", reuse=None, initializer=initializer):    
            testmodel = LSTM(dataset=dataset, is_training=False)
    
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
           
            ckpt = tf.train.get_checkpoint_state(config.checkpointpath) 
            if ckpt and ckpt.model_checkpoint_path:
                #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("nope, not like that") #TODO: er sollte es halt auch noch live haben.. kack tf.
                
            result = session.run([testmodel.logits], feed_dict={testmodel.input_data: dataset, testmodel.target: data_t})

            whatis = stats.mode(np.argmax(result[0], 1))[0][0]

            if doprint: 
                if not config.is_for_trump: 
                    print("-> Possible Rating: good movie" if whatis != 0 else "-> Possible Rating: bad movie")
                else:
                    print("-> Possible politial view: Trump-ee" if whatis != 0 else "-> Possible political view: non-trump-ee")
            
            return (whatis == 1)


class global_plot:
    def __init__(self, x_lim):
        plt.axis([0.9, x_lim+0.1, 0, 1.005])
        plt.ion()
        self.x_lim = x_lim
        self.current_x = 0
        self.trainvals = []
        self.testvals = []
        
    def update_plot(self, new_train, new_test):
        self.current_x += 1
        x = [elem+1 for elem in range(self.current_x)]
        self.trainvals.append(new_train)
        self.testvals.append(new_test)
        plt.axis([0.9, self.x_lim+0.1, 0, 1.005])
        savefig = plt.figure(1)
        plt.plot(x,self.trainvals,'b')
        plt.plot(x,self.testvals,'r')
        plt.pause(0.01)
        savefig.savefig(config.checkpointpath+"figure_dump.png")

        

def plot_test_and_train(dataset, amount_iterations):
    
    middlename = "_wordvecs" if config.use_w2v else ""
    pathname = config.checkpointpath+"ManyIterations/"
    
    for filename in os.listdir(pathname):
        os.remove(os.path.join(pathname, filename))
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTM(dataset=dataset, is_training=True)

        with tf.variable_scope("model", reuse=True, initializer=initializer): 
            testmodel = LSTM(dataset=dataset, is_training=False)
    
        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
  
        print("In this mode, we always create a model with fresh parameters.")
        init = tf.global_variables_initializer()
        init.run()
        iteration = 0
        
        if config.use_w2v:
            session.run(model.embedding.assign(dataset.wordvecs))
            session.run(testmodel.embedding.assign(dataset.wordvecs))
            print("Using the pre-trained word2vec")
        else:
            print("Not using the pre-trained word2vec")
           
        plot = global_plot(amount_iterations)
        print("Running",amount_iterations,"iterations.")
        test_accuracies = []
        
        for i in range(amount_iterations):
            train_accuracy = model.run_on(session, X_train, y_train, True, saver, iteration, i, amount_iterations, True)
            print("")
            test_accuracy = testmodel.run_on(session, X_test, y_test, False)
            print("Epoch: %d \t Train Accuracy: %.3f \t Testing Accuracy: %.3f" % (i + 1, train_accuracy, test_accuracy))          
  
            plot.update_plot(train_accuracy, test_accuracy)
            test_accuracies.append(test_accuracy)
            
        bestone = np.argmax(test_accuracies)+1    
        
        tokeep = "weights"+middlename+"_iteration"+str(bestone)
        for filename in os.listdir(pathname):
            if not tokeep in filename: 
                os.remove(os.path.join(pathname, filename))
        
        lines = ['model_checkpoint_path : "'+tokeep+'.ckpt"','all_model_checkpoint_paths : "'+tokeep+'.ckpt"', '#Iteration: "'+str(bestone)+'"']

        infile = open(pathname+"checkpoint", "w") #create a new file in writing mode,
        infile.write("\n".join(lines));  #and dump the content of our "lines" into it.
        infile.close()   

        print("Saved the best episode, #"+str(bestone)+", with accuracy",np.max(test_accuracies)) 
        if 0.35 < np.max(test_accuracies) < 0.65:
            print("I can tell you that it sucked. However, normally it does learn quite well. You may want to run it again, 85% distinction accuracy is possible.")
        
        if input("Shall I copy the best episode into "+config.checkpointpath+"? It may overwrite the current one in there.") in ('y','yes','Y','Yes','YES'):
         for filename in os.listdir(pathname):
            shutil.copy(pathname+filename, (pathname+filename).replace("ManyIterations/",""))
            
        return bestone


    
 
#==============================================================================

if input("Do you want to run the long version, which figures out the right amount of training etc automatically?") in ('y','yes','Y','Yes','YES'):
    print("Best training-set-result after",plot_test_and_train(moviedat, config.longruntrials),"iterations")
    validate(dataset=moviedat, path=config.checkpointpath+"ManyIterations/")
else:
    train_and_test(moviedat, config.TRAIN_STEPS)
    validate(dataset = moviedat, path = config.checkpointpath)

if config.is_for_trump:
    test_one_sample(moviedat, "I hate immigrants", True)
    test_one_sample(moviedat, "I hate Trump", True)
else:
    test_one_sample(moviedat, "I hated this movie. It sucks. The movie is bad, Worst movie ever. Bad Actors, bad everything.", True)
    test_one_sample(moviedat, "I loved this movie. It is awesome. The movie is good, best movie ever. good Actors, good everything.", True)


print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

#==============================================================================
#OK, now the Generative Model. Yay
#https://arxiv.org/pdf/1609.05473.pdf
#https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/seq-gan.md






