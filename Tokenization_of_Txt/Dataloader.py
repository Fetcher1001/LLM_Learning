import Dataset 
import random

class Dataloader():
    def __init__(self, dataset, batchsize=4, shuffle=False):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle

    def load(self):
        batches = []
        pairs = self.dataset
        if self.shuffle:
            random.shuffle(pairs)

        for i in range(0, len(pairs), self.batchsize):
            batches.append(pairs[i:i + self.batchsize])

        return batches

def load_dataset():
    Dset = Dataset(txt="LLM_Learning/Tokenization_of_Txt/the-verdict.txt", max_length=4, stride=1)
    inDset = Dset.pairs
    Dloader = Dataloader(inDset, batchsize=4)