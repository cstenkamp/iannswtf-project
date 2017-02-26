import numpy as np
from re import compile as _Re
import pickle
from pathlib import Path

def split_unicode_chrs(text):
    _unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split
    return [chr for chr in _unicode_chr_splitter(text) if chr]


class Dis_dataloader():
    
    
    
    def __init__(self):
        if Path("trumpdatweights/"+"dataset_mit_wordvecs.pkl").is_file():
            print("Dataset including word2vec found!")
            with open("trumpdatweights/"+'dataset_mit_wordvecs.pkl', 'rb') as input:
                self.dataset = pickle.load(input)       
                self.vocab_size = self.dataset.ohnum
        else:
            print("nope")
        
        

    def load_data_and_labels(self, positive_file, negative_file, howmany = 10000):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        positive_examples = []
        negative_examples = []


        amount = 0
        for currstring in self.dataset.trainreviews:
            if len(currstring) < 20:
                currstring.extend([0]*(20-len(currstring)))  #TODO!!! NULL IST UNK; NOT END!!!!!!!!!!!!!!
            positive_examples.append(currstring[:20])
            amount += 1
            if amount >= howmany:
                break



        amount = 0
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
                amount += 1
                if amount >= howmany:
                    break


        # Split by words
        x_text = positive_examples + negative_examples
        
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        x_text = np.array(x_text)
        y = np.array(y)
        return [x_text, y]




    def load_train_data(self, positive_file, negative_file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(positive_file, negative_file)
        
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        self.sequence_length = 20
        return [x_shuffled, y_shuffled]




    def load_test_data(self, positive_file, test_file, howmany=10000):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])


        amount = 0
        for currstring in self.dataset.trainreviews:
            if len(currstring) < 20:
                currstring.extend([0]*(20-len(currstring)))  #TODO!!! NULL IST UNK; NOT END!!!!!!!!!!!!!!
            test_examples.append(currstring[:20])
            test_labels.append([0, 1])
            amount += 1
            if amount >= howmany:
                break


        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]
    
    
    

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]








if __name__ == '__main__':
    positive_file = 'SeqGAN/save/real_data.txt'
    negative_file = 'SeqGAN/target_generate/generator_sample.txt'    
    dis_data_loader = Dis_dataloader()
    dis_x_train, dis_y_train = dis_data_loader.load_data_and_labels(positive_file, negative_file)
    print(dis_x_train.shape)