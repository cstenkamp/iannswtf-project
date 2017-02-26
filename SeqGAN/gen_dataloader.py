import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        
        
    def create_batches_from_data(self, dataset, howmany):
        self.token_stream = []
        amount = 0
        for currstring in self.dataset.trainreviews:
            if len(currstring) < 20:
                currstring.extend([0]*(20-len(currstring)))  #TODO!!! NULL IST UNK; NOT END!!!!!!!!!!!!!!
            self.token_stream.append(currstring[:20])
            amount += 1
            if howmany > 0 and amount >= howmany:
                break   
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0     
        
        
    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0