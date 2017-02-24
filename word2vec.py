# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:33:55 2017

@author: csten_000
"""
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import collections
import math
import random
import tensorflow as tf


def batch_buffer_append(config, dataset, firsttime=False):
    global dindex, permutations, currset
    if firsttime:
        lens = [len(dataset.traintargets) if config.w2v_usesets[0] else 0, len(dataset.testtargets) if config.w2v_usesets[1] else 0, len(dataset.validtargets) if config.w2v_usesets[2] else 0]
        currset = np.random.permutation([0]*lens[0]+[1]*lens[1]+[2]*lens[2])
        permutations = [np.random.permutation(i) for i in lens]
        dindex = [0,[0,0,0],0]  #dindex is: whichset, permuations[currset[dindex[0]]], index of that review
        return 
    else:
        whichset = currset[dindex[0]] 
        whereinset = permutations[whichset][(dindex[1][whichset])]
        if whichset == 0:   #wir haben ne zufällige reihenfolge, laut welcher aus train, test oder valid gezogen wird...
            currreview = dataset.trainreviews[whereinset]
        elif whichset == 1: #(allerdings 0 mal für set x falls set x nicht drankommen soll...)
            currreview = dataset.testreviews[whereinset]
        elif whichset == 2: #und innerhalb der 3 sets gibt es einen eigen fortlaufenden permutationsindex, sodass jedes element 1 mal dran kommt.
            currreview = dataset.validreviews[whereinset]
        toappend = currreview[dindex[2]]      
        dindex[2] += 1
        w2vsamplecount = 0    
        if dindex[2] >= len(currreview) or currreview[dindex[2]] == dataset.ohnum: #wenn der aktuelle review mit nummer x aus set y ende ist...
            dindex[2] = 0                                    #letzteres sollte der fall sein wenn wir am end-token sind..
            dindex[1][whichset] += 1
            dindex[0] += 1 #gehe zum nächstem review, das auch in einen anderem set sein kann
            w2vsamplecount = w2vsamplecount + 1
            if dindex[0] >= len(currset): #wenn du alle 3 sets durch hast..
                lens = [len(dataset.traintargets) if config.w2v_usesets[0] else 0, len(dataset.testtargets) if config.w2v_usesets[1] else 0, len(dataset.validtargets) if config.w2v_usesets[2] else 0]
                currset = np.random.permutation([0]*lens[0]+[1]*lens[1]+[2]*lens[2])
                permutations = [np.random.permutation(i) for i in lens]
                dindex = [0,[0,0,0],0] #...wird alles resettet.
                print("Once more through the entire dataset")
        return toappend, w2vsamplecount
    


# Function to generate a training batch for the skip-gram model.
def generate_batch(config, batch_size, num_skips, skip_window, dataset):
    w2vsamplecount = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        bufferappend, inc = batch_buffer_append(config, dataset)
        buffer.append(bufferappend)
        w2vsamplecount = w2vsamplecount + inc
    
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        bufferappend, inc = batch_buffer_append(config, dataset)
        buffer.append(bufferappend)
        w2vsamplecount = w2vsamplecount + inc
              
    return batch, labels, w2vsamplecount




# Step 4: Build and train a skip-gram model.
def perform_word2vec(config, dataset, print_example=False):
    w2vsamplecount = 0

    batch_buffer_append(config, dataset, True)
        
    if print_example:
        batch, labels, _ = generate_batch(config=config, batch_size=8, num_skips=2, skip_window=1, dataset=dataset, w2vsamplecount=0)
        for i in range(8):
          print(batch[i], dataset.uplook[batch[i]], '->', labels[i, 0], dataset.uplook[labels[i, 0]])
        #zwischen 2 generateten batches sind 1-2 wörter lücke, don't ask me why.
        batch_buffer_append(config, dataset, True)
      
    batch_size = 128
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
    
     
      with tf.device('/cpu:0'): #GPU implementation nonexistant yet
      
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([dataset.ohnum, config.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([dataset.ohnum, config.embedding_size],stddev=1.0 / math.sqrt(config.embedding_size)))
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
      valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
      similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
      # Add variable initializer.
      init = tf.global_variables_initializer()
    
    
    # Step 5: Begin training.
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print("Initialized")    
      average_loss = 0
      for step in xrange(config.num_steps_w2v):
        batch_inputs, batch_labels, inc = generate_batch(config, batch_size, num_skips, skip_window, dataset)
        w2vsamplecount = w2vsamplecount + inc
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
        if step % 20000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = dataset.uplook[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              try:
                  close_word = dataset.uplook[nearest[k]]
              except KeyError:
                  print("tried a non-possible key")
                  continue
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
            
      final_embeddings = normalized_embeddings.eval()
    return final_embeddings, w2vsamplecount
    


# Step 6: Visualize the embeddings.
def plot_tsne(final_embeddings, dataset, filename):
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
      plot_with_labels(low_dim_embs, labels, filename)
    except ImportError:
      print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

