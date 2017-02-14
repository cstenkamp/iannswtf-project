# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:22:43 2017

@author: csten_000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
from pathlib import Path
import tensorflow as tf
import random
import copy
import numpy as np
#np.set_printoptions(threshold=np.nan)
import collections
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.spatial.distance import cosine
import datetime

NAME = "train"
num_steps = 150001



class moviedata(object):
    def __init__(self, reviews, targets, lookup, uplook, count):
        self.reviews = reviews
        self.targets = targets
        self.lookup = lookup
        self.ohnum = count+1  #len(lookup)
        self.uplook = uplook
        
    def add_wordvectors(self, wordvecs):
        self.wordvecs = wordvecs


#==============================================================================

## first we create, save & load the words as indices.
def make_dataset():
    HOWMANY = 99999999
    allwords = {}
    counter = 1
    wordcount = 0
    
    with open("sets/"+NAME+".txt", encoding="utf8") as infile:
        string = []
        for line in infile: 
            words = line.split()
            for word in words:
                string.append(word)
                
    count = []
    count2 = []
    count.extend(collections.Counter(string).most_common(999999999))
    for elem in count:
        if elem[1] > 1:
            count2.append(elem[0])
        
    print("Most common words:")
    print(count[0:5])

    with open("sets/"+NAME+".txt", encoding="utf8") as infile:
        for line in infile:
            words = line.split()
            for word in words:
                if not word in allwords:
                    if word in count2: #words that only occur once don't count.
                        allwords[word] = wordcount
                        wordcount = wordcount +1
                    else:
                        allwords[word] = 0
            counter = counter + 1
            if counter > HOWMANY: 
                break
        #print(allwords)
    
    
    forreverse = copy.deepcopy(allwords)
    forreverse = { k:v for k, v in forreverse.items() if v > 0 }
    forreverse["<UNK>"] = 0
    reverse_dictionary = dict(zip(forreverse.values(), forreverse.keys()))
    reverse_dictionary[0] = "<UNK>"    
        
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
    moviedat = moviedata(ratings,ratetargets,allwords,reverse_dictionary,wordcount)
    return moviedat




# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, dataset):
    global dindex
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      global permutation
      buffer.append(dataset.reviews[permutation[(dindex[0] % len(permutation))]][dindex[1]])
      dindex[1] = dindex[1] + 1
      if dindex[1] >= len(dataset.reviews[permutation[(dindex[0] % len(permutation))]]):
          dindex[1] = 0
          dindex[0] = dindex[0] + 1
          permutation = np.random.permutation(len(dataset.targets))      
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(dataset.reviews[permutation[(dindex[0] % len(permutation))]][dindex[1]])
      dindex[1] = dindex[1] + 1
      if dindex[1] >= len(dataset.reviews[permutation[(dindex[0] % len(permutation))]]):
          dindex[1] = 0
          dindex[0] = dindex[0] + 1
          permutation = np.random.permutation(len(dataset.targets))      
    return batch, labels




# Step 4: Build and train a skip-gram model.
def perform_word2vec(dataset):
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # Number of negative examples to sample.
    
    graph = tf.Graph()
    with graph.as_default():
      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([dataset.ohnum, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([dataset.ohnum, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([dataset.ohnum]))
    
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=dataset.ohnum))
    
      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)
    
      # Add variable initializer.
      init = tf.global_variables_initializer()
    
    
    # Step 5: Begin training.
    global num_steps
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print("Initialized")    
      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window, dataset=dataset)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
    
        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": ", average_loss)
          average_loss = 0
    
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = dataset.uplook[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              close_word = dataset.uplook[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
            
      final_embeddings = normalized_embeddings.eval()
    return final_embeddings
    


# Step 6: Visualize the embeddings.
def plot_tsne(final_embeddings, dataset):
    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
          x, y = low_dim_embs[i, :]
          plt.scatter(x, y)
          plt.annotate(label,
                       xy=(x, y),
                       xytext=(5, 2),
                       textcoords='offset points',
                       ha='right',
                       va='bottom')
        plt.savefig(filename)
    try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt
      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
            
      labels = [dataset.uplook[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels)
    except ImportError:
      print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")



def closeones(dataset, indices):
    for i in indices:
        top_k = 5  # number of nearest neighbors
        dists = np.zeros(dataset.wordvecs.shape[0])
        for j in range(len(dataset.wordvecs)):
            dists[j] = cosine(dataset.wordvecs[i],dataset.wordvecs[j])
        dists[i] = float('inf')
        clos = np.argsort(dists)[:top_k]
        return [moviedat.uplook[i] for i in clos]

def printcloseones(dataset, word):
    print("Close to '",word.replace(" ",""),"': ",closeones(dataset,[dataset.lookup[word]]))
    

#==============================================================================

print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

if Path("./"+NAME+"ratings_mit_wordvecs.pkl").is_file():
    with open(NAME+'ratings_mit_wordvecs.pkl', 'rb') as input:
        moviedat = pickle.load(input)       

else:
    if Path("./"+NAME+"ratings_ohne_wordvecs.pkl").is_file():
        with open(NAME+'ratings_ohne_wordvecs.pkl', 'rb') as input:
            moviedat = pickle.load(input)  
            
    else:
        moviedat = make_dataset()
        print(""+str(moviedat.ohnum)+" different words.")
        rand = round(random.uniform(0,len(moviedat.targets)))
        print('Sample review', moviedat.reviews[rand][0:100], [moviedat.uplook[i] for i in moviedat.reviews[rand][0:100]])
        
        with open(NAME+'ratings_ohne_wordvecs.pkl', 'wb') as output:
            pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
       
    ## let's get to word2vec (https://www.tensorflow.org/tutorials/word2vec/)
    #TODO: CBOW statt skip-gram, da wir nen kleines dataset haben!
    permutation = np.random.permutation(len(moviedat.targets))
    dindex = [0,0]
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, dataset=moviedat)
    for i in range(8):
      print(batch[i], moviedat.uplook[batch[i]], '->', labels[i, 0], moviedat.uplook[labels[i, 0]])
    #zwischen 2 generateten batches sind 1-2 wörter lücke, don't ask me why.

    final_embeddings = perform_word2vec(moviedat)
    moviedat.add_wordvectors(final_embeddings)
    with open(NAME+'ratings_mit_wordvecs.pkl', 'wb') as output:
        pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)  


#plot_tsne(final_embeddings, moviedat)


printcloseones(moviedat, "woman")
printcloseones(moviedat, "<dot>")
printcloseones(moviedat, "movie")
printcloseones(moviedat, "his")
printcloseones(moviedat, "bad")
printcloseones(moviedat, "three")

print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))



#==============================================================================
#      
# #uniformly distributed unit cube    
# embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  
# #output weights and input weights for every word in the vocab  
# nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
# nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# #Placeholder for inputs for the skip-gram model graph. Each word is represented as an integer. Inputs: batch full of integers representing the source context words & the target words    
# train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# #look up the vector for each source word in the batch
# embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# #predict target-word using noise-constrastrive training objective:  Compute the NCE loss, using a sample of the negative labels each time.
# loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,num_sampled, vocabulary_size))
# #add the required loss nodes to compute gradients: Stochastic Gradient Descent Optimizer.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
# 
# #train the model: 
# for inputs, labels in generate_batch(...):
#   feed_dict = {training_inputs: inputs, training_labels: labels}
#   _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
#     
#==============================================================================


#def print_closeones(dataset, indices):
#    
#    graph = tf.Graph()
#    with graph.as_default():
#        with tf.device('/cpu:0'):
#
#            embeddings = tf.Variable(tf.random_uniform([dataset.ohnum, embedding_size], -1.0, 1.0))
#            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
#
#            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#            normalized_embeddings = embeddings / norm
#            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
#            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)            
#            
#            init = tf.global_variables_initializer()
#            
#            with tf.Session(graph=graph) as session:
#                # We must initialize all variables before we use them.
#                init.run()
#                print("Initialized")    
#                average_loss = 0
#              
#                batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, dataset=dataset)
#                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
#                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#            
#                sim = similarity.eval()
#
#
#                for i in indices:
#                    valid_word = dataset.uplook[i]
#                    top_k = 8  # number of nearest neighbors
#                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#                    log_str = "Nearest to %s:" % valid_word
#                    for k in xrange(top_k):
#                        close_word = dataset.uplook[nearest[k]]
#                        log_str = "%s %s," % (log_str, close_word)
#                    print(log_str)