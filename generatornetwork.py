#OK, now the Generative Model. Yay
#https://arxiv.org/pdf/1609.05473.pdf
#https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/seq-gan.md

###############################################################################
import time
import random
import numpy as np
import tensorflow as tf
#====own functions====
import file_functions

UNKINDEX = 0
EOSINDEX = 1


class LearnConfig(object):
  """Small config."""
  learning_rate = 1.0
  max_grad_norm = 5  #10 bei large
  num_steps = 20     #20 bei small, 35 bei large
  max_epoch = 4      #14 bei large
  max_max_epoch = 6  #55 bei large
  keep_prob = 0.9    #0.35 bei large
  lr_decay = 0.5     #1 / 1.15 bei large
  batch_size = 20


class TestGenConfig(LearnConfig):
  """Tiny config, for testing."""
  max_grad_norm = 1
  num_steps = 1
  max_epoch = 1
  max_max_epoch = 1
  batch_size = 1
  keep_prob = 1


###############################################################################

class LanguageModel(object):
    def __init__(self, mainconfig, dataset, is_training, config, is_generator=False):
        self.mainconfig = mainconfig

        self.batch_size = config.batch_size
        self.num_steps = config.num_steps   
        size = mainconfig.generatorhiddensize
        vocab_size = dataset.ohnum
    
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name="inputdata")
        if not is_generator:
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name="targets")      
            
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True), output_keep_prob=config.keep_prob)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            
        NUMLAYERS = 2   
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(NUMLAYERS)], state_is_tuple=True)        
    
        #since in our LSTM state_is_tuple, we have to deal with the initial state differently (see later)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
    
        with tf.device("/cpu:0"):
          embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
          inputs = tf.nn.embedding_lookup(embedding, self.input_data)
              
        if is_training:
          inputs = tf.nn.dropout(inputs, config.keep_prob)
    
        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self.initial_state) 
    
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
                          
        self.output_probs = tf.nn.softmax(logits)     
        self.final_state = state
    
        if is_generator:
            return         
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])],
                                                                  [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)], vocab_size)
           
        self.cost = cost = tf.reduce_sum(loss) / self.batch_size
    
        if not is_training:
          return
    
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)
    
    
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value, self.input_data: np.zeros([20,20]), self.targets: np.zeros([20,20])})


    def run_epoch(self, session, config, data, iterator, eval_op=None, verbose=False):
        epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(self.initial_state)
     
        fetches = {"cost": self.cost, "final_state": self.final_state,}
        if eval_op is not None:
          fetches["eval_op"] = eval_op
    
        for step, (x, y) in enumerate(iterator(data, self.batch_size, self.num_steps)):
          
            feed_dict = {}
            for i, (c, h) in enumerate(self.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
        
            feed_dict[self.input_data] = x
            feed_dict[self.targets] = y
    
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]
    
            costs += cost
            iters += self.num_steps
    
            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters), iters * config.batch_size / (time.time() - start_time)))
                
        return np.exp(costs / iters)


    def generate_text(self, session, config, howmany=1, nounk = True, lengmean = 0):

        def sample(a, temperature, nounk, noeos=False, eosprob=1):
            if nounk: a[UNKINDEX] = 0
            if noeos: a[EOSINDEX] = 0
            a[EOSINDEX] *= eosprob                       
            a = np.log(a) / temperature
            a = np.exp(a) / np.sum(np.exp(a))
            r = random.random() # range: [0,1)
            total = 0.0
            for i in range(len(a)):
                total += a[i]
                if total>r:
                    return i
            return len(a)-1         
        
        def softmax(x):
              scoreMatExp = np.exp(np.asarray(x))
              return scoreMatExp / scoreMatExp.sum(0)
        
        def thesample(a, letter, temperature=1.0, nounk=True, lengmean=0):
            atleast = 0 if letter < 2 else 1
            if lengmean == 0:
                return sample(a, temperature, nounk, eosprob=atleast)
            else:
                tmpa = list(range(lengmean))
                tmpb = list(range(lengmean))
                tmpb.reverse()
                tmpc = softmax(np.array(tmpa+tmpb)) #eg. [0.0058  0.0158  0.0430  0.1170  0.3182  0.3182  0.1170  0.0430  0.0158  0.0058]
                tmpc = [np.sum(tmpc[:i]) for i in range(len(tmpc))] #das ganze in kumuliert
                try:
                    if random.random() < tmpc[letter]:
                        a[EOSINDEX] *= 2
                except IndexError: #dann wären wir schon doppelt so lang wie der verlangte mean
                    a[EOSINDEX] *= 10
                return sample(a, temperature, nounk, eosprob=(letter/lengmean)*atleast) #ist am anfang sehr klein, 1 bei avglen, wird immer größer.
                
        state = session.run(self.initial_state)
        x = EOSINDEX # the id for '<eos>' from the training set #TODO: this.
        input = np.matrix([[x]])  # a 2D numpy matrix 
    
        feed_dict = {}
        for i, (c, h) in enumerate(self.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h        
    
        feed_dict[self.input_data] = input
  
        strings = []
        tmpstring = []
        sentencount = 0
        letter = 0
        newsentence = True
        while sentencount < howmany:
            if newsentence:
                  output_probs, state = session.run([self.output_probs, self.final_state], feed_dict)
                  newsentence = False
            else:
                  output_probs, state = session.run([self.output_probs, self.final_state],{self.input_data: input})          
            
            x = thesample(output_probs[0], letter, 0.9, nounk, lengmean)
     
            if x == EOSINDEX: #dann ist es eos #TODO: this
                strings.append(tmpstring)
                tmpstring = []
                sentencount += 1
                newsentence = True
                letter = 0
            else:
                tmpstring.append(x)
                letter += 1
            
            input = np.matrix([[x]]) 
          
        return strings




###############################################################################


def main_generate(mainconfig, dataset, howmany, nounk=True, avglen=0):
    config = TestGenConfig()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-mainconfig.allnetworkinitscale, mainconfig.allnetworkinitscale)
        with tf.name_scope("Generator"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = LanguageModel(mainconfig, dataset, is_training=False, config=config, is_generator=True) 
                with tf.Session() as session:
                    saver = tf.train.Saver() 
                    ckpt = tf.train.get_checkpoint_state(mainconfig.checkpointpath+"languagemodel/") 
                    assert ckpt and ckpt.model_checkpoint_path, "There must be a checkpoint!"
                    
                    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    saver.restore(session, ckpt.model_checkpoint_path)  
                    iteration = file_functions.read_iteration(path = mainconfig.checkpointpath+"languagemodel/")
                    print(iteration,"iterations ran already.") 
                    texts = m.generate_text(session, config, howmany, nounk, avglen)
                    return print_pretty(texts, dataset)
                        
                               
                        

def print_pretty(texts, dataset):
    strings = []
    for currtext in texts:
        string = ""
        for word in currtext:
            string += dataset.uplook[word] + " "
        string = dataset.prepareback(string)
        strings.append(string)
    return strings



def main(dataset, mainconfig, lmconfig):
    iterator = dataset.grammar_iterator
    
    train_data, valid_data, test_data, _ = dataset.return_all(only_positive = mainconfig.is_for_trump)
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-mainconfig.allnetworkinitscale, mainconfig.allnetworkinitscale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = LanguageModel(mainconfig, dataset, is_training=True, config=lmconfig)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = LanguageModel(mainconfig, dataset, is_training=False, config=lmconfig)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
              mtest = LanguageModel(mainconfig, dataset, is_training=False, config=TestGenConfig())

        with tf.Session() as session:
            saver = tf.train.Saver() 
            
            ckpt = tf.train.get_checkpoint_state(mainconfig.checkpointpath+"languagemodel/") 
            if ckpt and ckpt.model_checkpoint_path:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
                iteration = file_functions.read_iteration(path = mainconfig.checkpointpath+"languagemodel/")
                print(iteration,"iterations ran already.")
            else:
                print("Created model with fresh parameters.")
                init = tf.global_variables_initializer()
                init.run()
                iteration = 0
                
            print("Running for",lmconfig.max_max_epoch-iteration,"(further) iterations.")
            for i in range(lmconfig.max_max_epoch-iteration):
                lr_decay = lmconfig.lr_decay ** max(iteration+i+1 - lmconfig.max_epoch, 0.0)
                m.assign_lr(session, lmconfig.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = m.run_epoch(session, lmconfig, train_data, iterator, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = mvalid.run_epoch(session, lmconfig, valid_data, iterator)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                
                if mainconfig.checkpointpath+"languagemodel/" != "":
                    print("Saving model to %s." % mainconfig.checkpointpath+"languagemodel/")
                    saver.save(session, mainconfig.checkpointpath+"languagemodel/"+"genweights.ckpt")
                    file_functions.write_iteration(number = iteration+i+1, path=mainconfig.checkpointpath+"languagemodel/")
                    

            test_perplexity = mtest.run_epoch(session, lmconfig, test_data, iterator)
            print("Test Perplexity: %.3f" % test_perplexity)



###############################################################################

#if __name__ == "__main__":
#    config = LearnConfig()
#    main()
#    main_generate( reader.get_vocab("./Neuer Ordner/simple-examples/data/ptb.train.txt"), 40, nounk = True, avglen = 20)
    
