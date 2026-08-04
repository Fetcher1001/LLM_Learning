import Dataset 
import random

class Dataloader():
    def __init__(self, dataset, batchsize=4, shuffle=False):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle

    def load(self):
        batches = []
        pairs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(pairs)

        for i in range(0, len(pairs), self.batchsize):
            batches = [pairs[i:i + self.batchsize]]

        return batches

def load_dataset():
    Dset = Dataset
    inDset = Dset.pairs
    Dloader = Dataloader(inDset, batchsize=4)