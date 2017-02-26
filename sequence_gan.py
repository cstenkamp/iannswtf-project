import model
import numpy as np
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from target_lstm import TARGET_LSTM
from file_functions import read_iteration, write_iteration
import pickle

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 20
START_TOKEN = 0

PRE_EPOCH_NUM = 240 #240
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 64
##########################################################################################

TOTAL_BATCH = 800

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = 50 #50

positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000


##############################################################################################



def get_trainable_model(num_emb):
    return model.LSTM(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    end = time.time()
    print('Sample generation time:', (end - start))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            # buffer = u''.join([words[x] for x in poem]).encode('utf-8') + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader, saver):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch, saver)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def initialize_uninitialized_vars(session):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)
            


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 5000

    generator = get_trainable_model(vocab_size)
    with open('save/target_params.pkl', 'rb') as input:
        target_params = pickle.load(input, encoding='latin1') 
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)

    


    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    #saver = tf.train.Saver({"pretrain_updates": generator.pretrain_updates, "pretrain_loss": generator.pretrain_loss, "g_predictions": generator.g_predictions})    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("./") 
    if ckpt and ckpt.model_checkpoint_path:
        print("Reading preprocessing-model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        initialize_uninitialized_vars(sess)
        
        pretrainiters = pre_iteration = read_iteration("Preprocess-Iteration")
        print(pre_iteration,"preprocessing - iterations ran already.")
    else:
        print("Created preprocessing-model with fresh parameters.")
        init = tf.global_variables_initializer()
        init.run(session = sess)
        pretrainiters = pre_iteration = 0

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training for',PRE_EPOCH_NUM-pre_iteration,'further iterations (from',PRE_EPOCH_NUM,'because',pre_iteration,'are already done)')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM-pre_iteration):
        print('pre-train epoch:', pre_iteration+epoch+1)
        loss = pre_train_epoch(sess, generator, gen_data_loader, saver)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader) 
            print('pre-train epoch ', pre_iteration+epoch+1, 'test_loss ', test_loss)
            log.write(str(epoch) + ' ' + str(test_loss) + '\n')
        saver.save(sess, "./model.ckpt")
        pretrainiters = pre_iteration+epoch+1
        write_iteration(pretrainiters,"Preprocess-Iteration")

    generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    log.write('After pre-training:' + ' ' + str(test_loss) + '\n')

    generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')



    log.close()


if __name__ == '__main__':
    main()
