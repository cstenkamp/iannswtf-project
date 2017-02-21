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
import collections
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.spatial.distance import cosine
import datetime
import matplotlib.pyplot as plt
import copy
import time
from scipy import stats

w2vsamplecount = 0


class config(object):
    TRAINNAME = "train"
    TESTNAME = "test"
    VALIDATIONNAME = "validation"
    
    w2v_usesets = [True, True, True]
    embedding_size = 128
    num_steps_w2v = 200001 #198000 ist einmal durchs ganze dataset (falls nach word2vec gekürzt)
    
    use_w2v = True
    TRAIN_STEPS = 1
    batch_size = 32
    


class moviedata(object):
    def __init__(self, trainx, trainy, testx, testy, validx, validy, lookup, uplook, count):
        self.trainreviews = trainx
        self.traintargets = trainy
        self.testreviews = testx
        self.testtargets = testy
        self.validreviews = validx
        self.validtargets = validy
        self.lookup = lookup
        self.ohnum = count+1  #len(lookup)
        self.uplook = uplook
        
        
    def add_wordvectors(self, wordvecs):
        self.wordvecs = wordvecs


    def showstringlenghts(self, whichones, percentage, printstuff):
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
    
    #TODO: BUG: Das shortendata macht halbe wörter...! Es müsste nach worten brechen! :/
    def shortendata(self, whichones, percentage, lohnenderstring, printstuff):
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
            _ = self.lookup["<END>"]
        except KeyError:
            self.lookup["<END>"] = self.ohnum
            self.uplook[self.ohnum] = "<END>"
            self.ohnum += 1 #nur falls noch kein end-token drin ist+1!
        if hasattr(self, 'wordvecs'): #falls man es NACH word2vec ausführt
            self.wordvecs = np.append(self.wordvecs,np.transpose(np.transpose([[0]*config.embedding_size])),axis=0)
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
            return [moviedat.uplook[i] for i in clos]
    
    
    def printcloseones(self, word):
        print("Close to '",word.replace(" ",""),"': ",self.closeones([self.lookup[word]]))
    
                  
    def showarating(self,number):
        array = [moviedat.uplook[i] for i in self.trainreviews[number]]
        str = ' '.join(array)
        str = str.replace(" <comma>", ",")
        str = str.replace(" <colon>", ":")
        str = str.replace(" <openBracket>", "(")
        str = str.replace(" <closeBracket>", ")")
        str = str.replace(" <dot>", ".")
        str = str.replace(" <dots>", "...")
        str = str.replace(" <semicolon>", ";")
        str = str.replace("<quote>", '"')
        str = str.replace(" <question>", "?")
        str = str.replace(" <exclamation>", "!")
        str = str.replace(" <hyphen>  ","-")
        str = str.replace(" <END>", "")
        str = str.replace(" <SuperQuestion>", "???")
        str = str.replace(" <SuperExclamation>", "!!!")
        print(str)
        
#==============================================================================


def preparestring(string):
    str = copy.deepcopy(string)
    str = str.lower()
    str = str.replace(",", " <comma> ")
    str = str.replace(":", " <colon> ")
    str = str.replace("(", " <openBracket> ")
    str = str.replace(")", " <closeBracket> ")
    str = str.replace(".", " <dot> ")
    str = str.replace("...", " <dots> ")
    str = str.replace(";", " <semicolon> ")
    str = str.replace('"', " <quote> ")
    str = str.replace("?", " <question> ")
    str = str.replace("!", " <exclamation> ")
    str = str.replace("-", " <hyphen> ")
    str = str.replace("???", " <SuperQuestion> ")
    str = str.replace("!!!", " <SuperExclamation> ")
    while str.find("  ") > 0: str = str.replace("  "," ")
    if str.endswith(' '): str = str[:-1]
    return str
            
## first we create, save & load the words as indices.
def make_dataset(whichsets = [True, True, True]):
    allwords = {}
    wordcount = 1
    datasets = [config.TRAINNAME, config.TESTNAME, config.VALIDATIONNAME]
    
    #first we look how often each word occurs, to delete single occurences.
    for currset in range(3):
        if whichsets[currset]:
            with open("sets/"+datasets[currset]+".txt", encoding="utf8") as infile:
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
            with open("sets/"+datasets[currset]+".txt", encoding="utf8") as infile:
                for line in infile:
                    words = line.split()
                    for word in words:
                        if not word in allwords:
                            if word in count2: #words that only occur once don't count.
                                allwords[word] = wordcount
                                wordcount = wordcount +1
                            else:
                                allwords[word] = 0
    #print(allwords)            
    
    #the token for single occurences is "<UNK>"
    allwords["<UNK>"] = 0
    reverse_dictionary = dict(zip(allwords.values(), allwords.keys()))
        
    #now we make every ratings-string to an array of the respective numbers.
    ratings = [[],[],[]]
    for currset in range(3):
        if whichsets[currset]:        
            with open("sets/"+datasets[currset]+".txt", encoding="utf8") as infile:        
                for line in infile:
                    words = line.split()
                    currentrating = []
                    for word in words:
                        currentrating.append(allwords[word])
                    ratings[currset].append(currentrating)   
            
            
    #also, we make an array of the ratings
    ratetargets = [[],[],[]]
    for currset in range(3):
        if whichsets[currset]:     
            with open("sets/"+datasets[currset]+"-target.txt", encoding="utf8") as infile:
                for line in infile:
                    if int(line) < 5:
                        ratetargets[currset].append(0)
                    else:
                        ratetargets[currset].append(1)
            
    #we made a dataset! :)
    moviedat = moviedata(ratings[0], ratetargets[0],
                         ratings[1], ratetargets[1],
                         ratings[2], ratetargets[2],
                         allwords, reverse_dictionary, wordcount)
    
    return moviedat



        




# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, dataset):
    global dindex, permutations, currset, w2vsamplecount #dindex = [0,[0,0,0],0]
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        
      if currset[dindex[0]] == 0:   #wir haben ne zufällige reihenfolge, laut welcher aus train, test oder valid gezogen wird...
          temp = moviedat.trainreviews[dindex[1][currset[dindex[0]]]]
      elif currset[dindex[0]] == 1: #(allerdings 0 mal für set x falls set x nicht drankommen soll...)
          temp = moviedat.testreviews[dindex[1][currset[dindex[0]]]]
      elif currset[dindex[0]] == 2: #und innerhalb der 3 sets gibt es einen eigen fortlaufenden permutationsindex, sodass jedes element 1 mal dran kommt.
          temp = moviedat.validreviews[dindex[1][currset[dindex[0]]]]
      buffer.append(temp[dindex[2]])
      dindex[2] += 1
      if dindex[2] >= len(temp): #wenn der aktuelle review mit nummer x aus set y ende ist...
          dindex[2] = 0
          dindex[1][currset[dindex[0]]] += 1
          dindex[0] += 1 #gehe zum nächstem review, das auch in einen anderem set sein kann
          w2vsamplecount += 1
          if dindex[0] >= len(currset): #wenn du alle 3 sets durch hast..
              lens = [len(moviedat.traintargets) if config.w2v_usesets[0] else 0, len(moviedat.testtargets) if config.w2v_usesets[1] else 0, len(moviedat.validtargets) if config.w2v_usesets[2] else 0]
              currset = np.random.permutation([0]*lens[0]+[1]*lens[1]+[2]*lens[2])
              permutations = [np.random.permutation(i) for i in lens]
              dindex = [0,[0,0,0],0] #...wird alles resettet.
              print("Once more through the entire dataset")
              
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]

      if currset[dindex[0]] == 0:   #wir haben ne zufällige reihenfolge, laut welcher aus train, test oder valid gezogen wird...
          temp = moviedat.trainreviews[dindex[1][currset[dindex[0]]]]
      elif currset[dindex[0]] == 1: #(allerdings 0 mal für set x falls set x nicht drankommen soll...)
          temp = moviedat.testreviews[dindex[1][currset[dindex[0]]]]
      elif currset[dindex[0]] == 2: #und innerhalb der 3 sets gibt es einen eigen fortlaufenden permutationsindex, sodass jedes element 1 mal dran kommt.
          temp = moviedat.validreviews[dindex[1][currset[dindex[0]]]]
      buffer.append(temp[dindex[2]])
      dindex[2] += 1
      if dindex[2] >= len(temp): #wenn der aktuelle review mit nummer x aus set y ende ist...
          dindex[2] = 0
          dindex[1][currset[dindex[0]]] += 1
          dindex[0] += 1 #gehe zum nächstem review, das auch in einen anderem set sein kann
          w2vsamplecount += 1
          if dindex[0] >= len(currset): #wenn du alle 3 sets durch hast..
              lens = [len(moviedat.traintargets) if config.w2v_usesets[0] else 0, len(moviedat.testtargets) if config.w2v_usesets[1] else 0, len(moviedat.validtargets) if config.w2v_usesets[2] else 0]
              currset = np.random.permutation([0]*lens[0]+[1]*lens[1]+[2]*lens[2])
              permutations = [np.random.permutation(i) for i in lens]
              dindex = [0,[0,0,0],0] #...wird alles resettet.
              print("Once more through the entire dataset")
                    
    return batch, labels




# Step 4: Build and train a skip-gram model.
def perform_word2vec(dataset):
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
        if step % 20000 == 0:
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





def to_one_hot(y):
    y_one_hot = []
    for row in y:
        if row == 0:
            y_one_hot.append([1.0, 0.0])
        else:
            y_one_hot.append([0.0, 1.0])
    return np.array([np.array(row) for row in y_one_hot])

#==============================================================================
def read_iteration():
    try:
        with open("checkpoint", encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if line[:11] == "#Iteration:":
                     iterations = int(line[line.find('"'):][1:-1])
                     return iterations
            return 0
    except FileNotFoundError:
        return 0

def increase_iteration():
    write_iteration(read_iteration()+1)
    
def write_iteration(number):
    try:
        lines = []
        with open("checkpoint", encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if not line[:11] == "#Iteration:":
                    lines.append(line)
            lines.append('#Iteration: "'+str(number)+'"')
    except FileNotFoundError:
        lines.append('#Iteration: "'+str(number)+'"')
    infile = open("checkpoint", "w")
    infile.write("\n".join(lines));        
    infile.close()            
#==============================================================================


print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
print('Loading data...')

if Path("./ratings_mit_wordvecs.pkl").is_file():
    print("Dataset including word2vec found!")
    with open('ratings_mit_wordvecs.pkl', 'rb') as input:
        moviedat = pickle.load(input)       

else:
    if Path("./ratings_ohne_wordvecs.pkl").is_file():
        print("dataset without word2vec found.")
        with open('ratings_ohne_wordvecs.pkl', 'rb') as input:
            moviedat = pickle.load(input)  
        print(moviedat.ohnum," different words.")
        moviedat.ohnum += 1 #TODO: DAS HIER MUSS RAUS SOBALD DAS DATASET NOCHMAL ERSTELLT WIRD!!
            
    else:
        print("No dataset found! Creating new...")
        moviedat = make_dataset(config.w2v_usesets)
        #print("Shortening to", moviedat.shortendata([True, True, True], .75, 40, True))
        print(""+str(moviedat.ohnum)+" different words.")
        rand = round(random.uniform(0,len(moviedat.traintargets)))
        print('Sample review', moviedat.trainreviews[rand][0:100], [moviedat.uplook[i] for i in moviedat.trainreviews[rand][0:100]])
        
        with open('ratings_ohne_wordvecs.pkl', 'wb') as output:
            pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
            print('Saved the dataset as Pickle-File')
       
    ## let's get to word2vec (https://www.tensorflow.org/tutorials/word2vec/)
    #TODO: CBOW statt skip-gram, da wir nen kleines dataset haben!
    print("Starting word2vec...")
    
    lens = [len(moviedat.traintargets) if config.w2v_usesets[0] else 0, len(moviedat.testtargets) if config.w2v_usesets[1] else 0, len(moviedat.validtargets) if config.w2v_usesets[2] else 0]
    currset = np.random.permutation([0]*lens[0]+[1]*lens[1]+[2]*lens[2])
    permutations = [np.random.permutation(i) for i in lens]
    dindex = [0,[0,0,0],0]  #dindex is: whichset, permuations[currset[dindex[0]]], index of that review
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, dataset=moviedat)
    for i in range(8):
      print(batch[i], moviedat.uplook[batch[i]], '->', labels[i, 0], moviedat.uplook[labels[i, 0]])
    #zwischen 2 generateten batches sind 1-2 wörter lücke, don't ask me why.

    final_embeddings = perform_word2vec(moviedat)
    moviedat.add_wordvectors(final_embeddings)
    with open('ratings_mit_wordvecs.pkl', 'wb') as output:
        pickle.dump(moviedat, output, pickle.HIGHEST_PROTOCOL)
    print("Saved word2vec-Results.")
    print("Word2vec ran through ",w2vsamplecount," different reviews.")
    moviedat.printcloseones("woman")
    moviedat.printcloseones("<dot>")
    moviedat.printcloseones("movie")
    moviedat.printcloseones("his")
    moviedat.printcloseones("bad")
    moviedat.printcloseones("three")


#plot_tsne(moviedat.wordvecs, moviedat)

print('Data loaded.')

print("Shortening to", moviedat.shortendata([True, True, True], .75, 40, False))
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
    def __init__(self, is_training):
    
        self.input_data = tf.placeholder(tf.int32, [config.batch_size, moviedat.maxlenstring], name="input_x")
        self.target = tf.placeholder(tf.float32, [config.batch_size, 2], name="input_t") #2 = n_classes
    
        #non-stateful LSTM   #128 ist hidden_size (=#Vectors???)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=0.0, state_is_tuple=True)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
            
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
        initial_state = cell.zero_state(config.batch_size, tf.float32)
            
        if config.use_w2v:
#            embedding = tf.Variable(moviedat.wordvecs, name="embedding", dtype=tf.float32)
#            inputs = tf.nn.embedding_lookup(embedding, self.input_data)     
#            W = tf.Variable(tf.constant(0.0, shape=[moviedat.ohnum, config.embedding_size]), trainable=False, name="W")
#            embedding_placeholder = tf.placeholder(tf.float32, [moviedat.ohnum, config.embedding_size])
#            embedding_init = W.assign(embedding_placeholder)
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                embedding = tf.Variable(tf.random_uniform([moviedat.ohnum, config.embedding_size], -1.0, 1.0), trainable = False, name="embedding")
                self.embedding = embedding
                inputs = tf.nn.embedding_lookup(embedding, self.input_data, name="embeddings")
        else:            
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [moviedat.ohnum+1, 128], dtype=tf.float32)
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
                savename = "./movierateweights_wordvecs.ckpt"
            else:
                savename = "./movierateweights.ckpt"
                
            saver.save(session, savename)
            #TODO: beim SaveALot-Modus sollte er das jetzt re-namen in "_iterationx"
            
            time.sleep(0.1)
            write_iteration(iteration+epoch+1)
        
        return accuracy
    

def initialize_uninitialized_vars(session):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)
            


def train_and_test(amount_iterations):
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
     #TODO: abfragen ob er lernen, applien oder beides will (oder vie-zeit-modus)
     #viel-zeit-modus: wo er train und test accuracy live errechnet und direkt plottet und man sich das beste aussuchen kann
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTM(is_training=True)
    
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
           
            ckpt = tf.train.get_checkpoint_state("./") #TODO: da unterscheidet er noch nicht zwischen mit und ohne w2v..
            if ckpt and ckpt.model_checkpoint_path:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
                iteration = read_iteration()
                print(iteration,"iterations ran already.")
            else:
                print("Created model with fresh parameters.")
                init = tf.global_variables_initializer()
                init.run()
                iteration = 0
            
            if config.use_w2v:
                session.run(model.embedding.assign(moviedat.wordvecs))
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
            testmodel = LSTM(is_training=False)
            
            if config.use_w2v:
                session.run(testmodel.embedding.assign(moviedat.wordvecs))
                print("Using the pre-trained word2vec")
            else:
                print("Not using the pre-trained word2vec")
                
                
            test_accuracy = testmodel.run_on(session, X_test, y_test, False)
            print("Testing Set Accuracy: %.3f" % (test_accuracy))          


train_and_test(config.TRAIN_STEPS)



def test_one_sample(string, doprint=False):
    if doprint: print(string)
    dataset = [moviedat.lookup[i] if i in moviedat.lookup.keys() else moviedat.lookup["<UNK>"] for i in preparestring(string).split(" ")] #oh damn, für solche einzeiler liebe ich python.
    dataset = dataset + [0]*(moviedat.maxlenstring-len(dataset))
    dataset = [dataset]*config.batch_size 
    data_t = to_one_hot([0]*config.batch_size)
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
        with tf.variable_scope("model", reuse=None, initializer=initializer):    
            testmodel = LSTM(is_training=False)
    
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
           
            ckpt = tf.train.get_checkpoint_state("./") 
            if ckpt and ckpt.model_checkpoint_path:
                #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("nope, not like that") #TODO: er sollte es halt auch noch live haben.. kack tf.
                
            result = session.run([testmodel.logits], feed_dict={testmodel.input_data: dataset, testmodel.target: data_t})

            whatis = stats.mode(np.argmax(result[0], 1))[0][0]

            if doprint: print("good movie" if whatis != 0 else "bad movie")
            
            return (whatis == 0)

test_one_sample("I hated this movie. It sucks.", True)


class global_plot:
    def __init__(self, x_lim):
        self.x_lim = x_lim
        self.fig = plt.figure()
        plt.ion()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_xlim([0, self.x_lim])
        self.ax1.set_ylim([0, 1])
        self.current_x = 0
        self.trainvals = []
        self.testvlals = []
        
    def update_plot(self, new_train, new_test):
        self.current_x += 1
        x = range(self.current_x)
        self.trainvals.append(new_train)
        self.testvals.append(new_test)
        self.ax1.clear()
        self.ax1.set_xlim([0, self.x_lim])
        self.ax1.set_ylim([0, 1])
        self.ax1.plot(x,self.trainvals,'b')
        self.ax1.plot(x,self.testvals,'r')
        self.fig.canvas.draw()


def plot_test_and_train(amount_iterations):
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTM(is_training=True)

        with tf.variable_scope("model", reuse=True, initializer=initializer): 
            testmodel = LSTM(is_training=False)
    
        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
  
        print("In this mode, we always create a model with fresh parameters.")
        init = tf.global_variables_initializer()
        init.run()
        iteration = 0
        
        if config.use_w2v:
            session.run(model.embedding.assign(moviedat.wordvecs))
            session.run(testmodel.embedding.assign(moviedat.wordvecs))
            print("Using the pre-trained word2vec")
        else:
            print("Not using the pre-trained word2vec")
           
        plot = global_plot(amount_iterations)
        print("Running",amount_iterations,"iterations.")
        
        for i in range(amount_iterations):
            train_accuracy = model.run_on(session, X_train, y_train, True, saver, iteration, i, amount_iterations, True)
            test_accuracy = testmodel.run_on(session, X_test, y_test, False)
            print("Epoch: %d \t Train Accuracy: %.3f \t Testing Accuracy: %.3f" % (i + 1, train_accuracy, test_accuracy))          
  
            plot.update_plot(train_accuracy, test_accuracy)
    
    
#def prepare_checkpoint():
    #TODO: was das hier macht die "checkpoint"-datei zu ändern, in entweder mit oder ohne own word2vec


#==============================================================================



#W = tf.Variable(tf.constant(0.0, shape=[moviedat.ohnum+1, config.embedding_size]), trainable=False, name="W")
#
#embedding_placeholder = tf.placeholder(tf.float32, [moviedat.ohnum+1, config.embedding_size])
#embedding_init = W.assign(embedding_placeholder)
#
#
#sess = tf.Session()
#
#sess.run(embedding_init, feed_dict={embedding_placeholder: moviedat.wordvecs})


#inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
#targets = tf.placeholder(tf.float32, [batch_size, n_classes])

    


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

