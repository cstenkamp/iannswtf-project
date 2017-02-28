import time
import random
import numpy as np
import tensorflow as tf
import copy

import reader


UNKINDEX = 1
EOSINDEX = 2

data_path = "./Neuer Ordner/simple-examples/data"
save_path = "./Neuer Ordner/save/"

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 6
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class SmallGenConfig(object):
  """Small config. for generation"""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 1 # this is the main difference
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000  


config = SmallConfig()  #TestConfig()



###############################################################################

class LanguageModel(object):
    def __init__(self, is_training, config, is_generator=False):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps   
        size = config.hidden_size
        vocab_size = config.vocab_size
    
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        if not is_generator:
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])      
            
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True), output_keep_prob=config.keep_prob)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(config.num_layers)], state_is_tuple=True)        
    
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
        state = session.run(self.initial_state, {self.input_data: np.zeros([20,20]), self.targets: np.zeros([20,20])})
     
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
            if lengmean == 0:
                return sample(a, temperature, nounk)
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
                return sample(a, temperature, nounk, eosprob=letter/lengmean) #ist am anfang sehr klein, 1 bei avglen, wird immer größer.
                
        
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


def main_generate(savepath, vocab, howmany, nounk=True, avglen=0):
    graph = tf.Graph()
    config = SmallGenConfig()
    with graph.as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.name_scope("Generator"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = LanguageModel(is_training=False, config=config, is_generator=True) 
                with tf.Session() as session:
                    saver = tf.train.Saver() 
                    ckpt = tf.train.get_checkpoint_state(savepath) 
                    if ckpt and ckpt.model_checkpoint_path:
                        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                        saver.restore(session, ckpt.model_checkpoint_path)                
                        
                        texts = m.generate_text(session, config, howmany, nounk, avglen)
                        
                        print(print_pretty(texts, vocab))
                        

def print_pretty(texts, vocab):
    string = ""
    for currtext in texts:
        print(len(currtext))
        for word in currtext:
            string += vocab[word] + " "
        string += "\n\n"
    return string



def main():
    graph = tf.Graph()

    raw_data = reader.ptb_raw_data(data_path)
    #raw_data = dataset.return_all()
    train_data, valid_data, test_data, _ = raw_data  

    eval_config = copy.deepcopy(config)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = LanguageModel(is_training=True, config=config)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = LanguageModel(is_training=False, config=config)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
              mtest = LanguageModel(is_training=False, config=eval_config)

        sv = tf.train.Supervisor(logdir=save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                iterator = reader.ptb_iterator
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = m.run_epoch(session, config, train_data, iterator, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = mvalid.run_epoch(session, mvalid, config, valid_data, iterator)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = mtest.run_epoch(session, config, test_data, iterator)
            print("Test Perplexity: %.3f" % test_perplexity)

            if save_path != "":
                print("Saving model to %s." % save_path)
                sv.saver.save(session, save_path, global_step=sv.global_step)


###############################################################################

if __name__ == "__main__":
    main()
    #main_generate(save_path, reader.get_vocab("./Neuer Ordner/simple-examples/data/ptb.train.txt"), 40, nounk = True, avglen = 20)
    
